import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicExpert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicExpert, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class BasicMOE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super(BasicMOE, self).__init__()
        self.experts = nn.ModuleList(
            [BasicExpert(input_dim, output_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # x: (batch_size, input_dim)
        # gate_output: (batch_size, num_experts)
        gate_output = F.softmax(self.gate(x), dim=-1)
        # expert_outputs: (batch_size, num_experts, output_dim)
        expert_outputs = torch.stack([expert(x)
                                     for expert in self.experts], dim=1)
        # gate_output: (batch_size, 1, num_experts)
        gate_output = gate_output.unsqueeze(1)
        # output: (batch_size, 1, output_dim)
        output = gate_output @ expert_outputs
        # (batch_size, output_dim)
        output = output.squeeze(1)
        return output


def test_basic_moe():
    input_dim = 512
    output_dim = 128
    num_experts = 4
    batch_size = 4

    model = BasicMOE(input_dim, output_dim, num_experts)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


class MOEConfig:
    def __init__(self, hidden_dim, expert_number, top_k, shared_experts_number):
        self.hidden_dim = hidden_dim
        self.expert_number = expert_number
        self.top_k = top_k
        self.shared_experts_number = shared_experts_number


class MOERouter(nn.Module):
    def __init__(self, config: MOEConfig):
        super(MOERouter, self).__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_dim, config.expert_number)
        self.expert_number = config.expert_number
        self.top_k = config.top_k

    def forward(self, x):
        # (batch_size * seq_len, expert_number)
        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits, dim=-1)
        # (batch_size * seq_len, top_k)
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        top_k_probs = top_k_probs.to(dtype=x.dtype)
        # (batch_size * seq_len, top_k, expert_number)
        # batch_size * seq_len 代表每个样本的所有 token
        # top_k 代表每个 token 选择的前 k 个专家在 expert_number 中的位置
        expert_mask = F.one_hot(top_k_indices, num_classes=self.expert_number)
        expert_mask = expert_mask.permute(0, 2, 1)
        # router_logits: (batch_size * seq_len, expert_number)
        # top_k_probs: (batch_size * seq_len, top_k)
        # top_k_indices: (batch_size * seq_len, top_k)
        # expert_mask: (expert_number, top_k, batch_size * seq_len)
        return router_logits, top_k_probs, top_k_indices, expert_mask


class SparseMOE(nn.Module):
    def __init__(self, config: MOEConfig):
        super(SparseMOE, self).__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [BasicExpert(config.hidden_dim, config.hidden_dim) for _ in range(config.expert_number)])
        self.router = MOERouter(config)

    def forward(self, x):
        # (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape
        # (batch_size * seq_len, hidden_dim)
        x = x.view(-1, hidden_dim)
        router_logits, top_k_probs, top_k_indices, expert_mask = self.router(x)
        out = torch.zeros(
            (batch_size * seq_len, hidden_dim), device=x.device, dtype=x.dtype)
        for i in range(self.config.expert_number):
            expert_layer = self.experts[i]
            # expert_mask: (expert_number, top_k, batch_size * seq_len)
            # (top_k, batch_size * seq_len)
            current_expert_mask = expert_mask[i]
            # idx 为行索引，代表选中的 topk 专家位置
            # top_x 为列索引，代表 token 的位置
            idx, top_x = torch.where(current_expert_mask)
            # current_x: (selected_token_num, hidden_dim)
            current_x = x[idx]
            current_x = expert_layer(current_x)
            # current_logits: selected_token_num
            current_logits = top_k_probs[top_x, idx]
            # (selected_token_num, 1)
            current_logits = current_logits.unsqueeze(-1)
            # (selected_token_num, hidden_dim)
            current_state = current_x * current_logits
            out.index_add_(0, top_x, current_state)
        out = out.view(batch_size, seq_len, hidden_dim)
        return out, router_logits


def test_sparse_moe():
    x = torch.randn(2, 4, 16)  # (batch_size, seq_len, hidden_dim)
    config = MOEConfig(hidden_dim=16, expert_number=2,
                       top_k=2, shared_experts_number=4)
    model = SparseMOE(config)
    output, router_logits = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


class SharedExpertMOE(nn.Module):
    def __init__(self, config: MOEConfig):
        super(SharedExpertMOE, self).__init__()
        self.config = config
        self.routed_experts_moe = SparseMOE(config)
        self.shared_experts = nn.ModuleList(
            [BasicExpert(config.hidden_dim, config.hidden_dim)
             for _ in range(config.shared_experts_number)]
        )

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        # (batch_size, seq_len, hidden_dim)
        shared_expert_out_list = [expert(x)
                                  for expert in self.shared_experts]
        # (shared_experts_number, batch_size, seq_len, hidden_dim)
        shared_expert_out = torch.stack(shared_expert_out_list, dim=0)
        # (batch_size, seq_len, hidden_dim)
        shared_expert_out = shared_expert_out.sum(dim=0, keepdim=False)
        sparse_moe_out, router_logits = self.routed_experts_moe(x)
        out = sparse_moe_out + shared_expert_out
        return out, router_logits


def test_shared_expert_moe():
    x = torch.randn(2, 4, 16)  # (batch_size, seq_len, hidden_dim)
    config = MOEConfig(hidden_dim=16, expert_number=2,
                       top_k=2, shared_experts_number=4)
    model = SharedExpertMOE(config)
    output, router_logits = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    # test_basic_moe()
    # test_sparse_moe()
    test_shared_expert_moe()
