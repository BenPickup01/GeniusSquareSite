import torch
import torch.nn as nn

class MultiHeadValueNet(nn.Module):
    def __init__(self, in_channels=6, board_size=6, conv_out_channels=8):
        super().__init__()
        self.board_size = board_size
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels

        # First head processes the first 3 channels together
        self.head0 = nn.Sequential(
            nn.Conv2d(3, conv_out_channels, kernel_size=3, padding=1),  # (B, conv_out, 6, 6)
            nn.ReLU(),
            nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()  # (B, conv_out * 6 * 6)
        )
        # Then one head per remaining channel
        self.other_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, conv_out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten()
            ) for _ in range(max(0, in_channels - 3))
        ])

        # Total number of heads
        num_heads = 1 + max(0, in_channels - 3)
        total_feature_dim = conv_out_channels * board_size * board_size * num_heads

        # MLP to combine all head features
        self.mlp = nn.Sequential(
            nn.Linear(total_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),  # deeper layers
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):  # x: (B, C, 6, 6)
        # First head on channels [0,1,2]
        outputs = []
        first = x[:, :3, :, :]
        outputs.append(self.head0(first))
        # Remaining single-channel heads
        for i, head in enumerate(self.other_heads, start=3):
            xi = x[:, i:i+1, :, :]
            outputs.append(head(xi))

        # Concatenate and run through MLP
        combined = torch.cat(outputs, dim=1)
        return self.mlp(combined).squeeze(-1)

class SingleHeadValueNet(nn.Module):
    def __init__(
        self,
        board_size: int,
        spatial_channel_indices: list,
        score_channel_indices: list,
        conv_out_channels: int = 8,
        score_mlp_hidden: int = 32,
        score_embed_dim: int = 16,
    ):
        """
        Args:
          board_size: height/width of the square board (e.g. 6)
          spatial_channel_indices: list of channel indices for spatial conv input
          score_channel_indices: list of channel indices for constant score planes
        """
        super().__init__()
        self.board_size = board_size
        self.spatial_indices = spatial_channel_indices
        self.score_indices = score_channel_indices

        # Single convolutional head processing all spatial channels together
        in_spatial = len(spatial_channel_indices)
        self.conv_head = nn.Sequential(
            nn.Conv2d(in_spatial, conv_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # MLP to embed score scalars
        n_scores = len(score_channel_indices)
        self.score_mlp = nn.Sequential(
            nn.Linear(n_scores, score_mlp_hidden),
            nn.ReLU(),
            nn.Linear(score_mlp_hidden, score_embed_dim),
            nn.ReLU()
        )

        # Final deep MLP to combine conv features + score embeddings
        total_conv_feat = conv_out_channels * board_size * board_size
        total_dim = total_conv_feat + score_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        Channels = spatial + score
        """
        # 1) Select and process all spatial channels through one conv head
        spatial_in = x[:, self.spatial_indices, :, :]  # (B, len(spatial), H, W)
        conv_feat = self.conv_head(spatial_in)         # (B, conv_out * H * W)

        # 2) Extract score scalars
        scores = [x[:, idx, 0, 0].unsqueeze(1) for idx in self.score_indices]
        scores = torch.cat(scores, dim=1)                # (B, n_scores)
        score_feat = self.score_mlp(scores)              # (B, score_embed_dim)

        # 3) Concatenate and predict
        combined = torch.cat([conv_feat, score_feat], dim=1)  # (B, total_dim)
        return self.mlp(combined).squeeze(-1)                 # (B,)

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    A simple residual block with two convolutional layers and batch normalization.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = nn.ReLU(inplace=True)(out)
        return out


class AttentionValueNet(nn.Module):
    """
    A value network that combines residual convolutional blocks with a lightweight
    transformer encoder over spatial tokens to capture both local and global board structure.
    """
    def __init__(
        self,
        in_channels: int = 5,
        board_size: int = 6,
        conv_channels: int = 32,
        num_res_blocks: int = 2,
        attn_embed_dim: int = 32,
        num_heads: int = 4,
        num_attn_layers: int = 2,
        mlp_hidden_dim: int = 64,
    ):
        super().__init__()
        self.board_size = board_size
        # Initial convolutional stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Residual backbone
        self.res_layers = nn.Sequential(*[ResidualBlock(conv_channels) for _ in range(num_res_blocks)])

        # Positional embeddings for each cell
        seq_len = board_size * board_size
        self.pos_embed = nn.Parameter(torch.randn(seq_len, conv_channels))

        # Transformer encoder over spatial tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_channels,
            nhead=num_heads,
            dim_feedforward=conv_channels * 4,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_attn_layers)

        # Final value head
        self.value_head = nn.Sequential(
            nn.Linear(conv_channels, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W)
        B = x.shape[0]
        out = self.stem(x)               # (B, C, H, W)
        out = self.res_layers(out)       # (B, C, H, W)
        # Flatten spatial dims to sequence
        C = out.size(1)
        tokens = out.view(B, C, -1).permute(2, 0, 1)  # (S, B, C), where S = H*W
        # Add positional embeddings
        tokens = tokens + self.pos_embed.unsqueeze(1)
        # Transformer encoding
        attn_out = self.transformer(tokens)           # (S, B, C)
        # Pool across tokens
        pooled = attn_out.mean(dim=0)                # (B, C)
        # Final MLP to scalar value
        value = self.value_head(pooled).squeeze(-1)  # (B,)
        return value

class DualHeadAttentionValueNet(nn.Module):
    """
    A value network with two heads: one processing spatial channels via CNN + Transformer,
    and another embedding scalar score channels, then combining for final value.
    """
    def __init__(
        self,
        spatial_channel_indices: list,
        score_channel_indices: list,
        board_size: int = 6,
        conv_channels: int = 32,
        num_res_blocks: int = 2,
        num_heads: int = 4,
        num_attn_layers: int = 2,
        score_mlp_hidden: int = 32,
        mlp_hidden_dim: int = 128,
    ):
        super().__init__()
        self.board_size = board_size
        self.spatial_indices = spatial_channel_indices
        self.score_indices = score_channel_indices

        # Spatial CNN + Transformer head
        in_spatial = len(self.spatial_indices)
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(in_spatial, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.spatial_res = nn.Sequential(*[ResidualBlock(conv_channels) for _ in range(num_res_blocks)])
        # Positional embeddings
        seq_len = board_size * board_size
        self.pos_embed = nn.Parameter(torch.randn(seq_len, conv_channels))
        # Transformer encoder (batch_first)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv_channels,
            nhead=num_heads,
            dim_feedforward=conv_channels * 4,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_attn_layers)

        # Score embedding head
        self.score_mlp = nn.Sequential(
            nn.Linear(len(self.score_indices), score_mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(score_mlp_hidden, score_mlp_hidden),
            nn.ReLU(inplace=True),
        )

        # Final combining MLP
        total_dim = conv_channels + score_mlp_hidden
        self.comb_mlp = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # 1) Spatial head
        spatial_in = x[:, self.spatial_indices, :, :]                # (B, S, H, W)
        out = self.spatial_stem(spatial_in)                          # (B, conv, H, W)
        out = self.spatial_res(out)                                  # (B, conv, H, W)
        tokens = out.view(B, out.size(1), -1).permute(0, 2, 1)       # (B, S, conv)
        tokens = tokens + self.pos_embed.unsqueeze(0)                # add pos embed
        attn = self.transformer(tokens)                              # (B, S, conv)
        spatial_feat = attn.mean(dim=1)                              # (B, conv)

        # 2) Score head
        scores = [x[:, idx, 0, 0].unsqueeze(1) for idx in self.score_indices]
        scores = torch.cat(scores, dim=1)                            # (B, num_scores)
        score_feat = self.score_mlp(scores)                          # (B, score_mlp_hidden)

        # 3) Combine and predict
        combined = torch.cat([spatial_feat, score_feat], dim=1)      # (B, total_dim)
        value = self.comb_mlp(combined).squeeze(-1)                   # (B,)
        return value
    
class UpdatedSingleHead(nn.Module):
    """
    A value network that processes spatial channels via convolution,
    then concatenates raw score scalars directly into the final MLP.
    """
    def __init__(
        self,
        board_size: int,
        spatial_channel_indices: list,
        score_channel_indices: list,
        conv_out_channels: int = 32,
    ):
        super().__init__()
        self.board_size = board_size
        self.spatial_indices = spatial_channel_indices
        self.score_indices = score_channel_indices

        # Convolutional head for spatial channels
        in_spatial = len(self.spatial_indices)
        # Use dilation in second conv to enlarge receptive field (covers up to 5×5 area)
        self.conv_head = nn.Sequential(
            nn.Conv2d(in_spatial, conv_out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        # Final MLP combining conv features + raw scores
        total_conv_feat = conv_out_channels * board_size * board_size
        total_dim = total_conv_feat + len(self.score_indices)
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # 1) Spatial conv features
        spatial_in = x[:, self.spatial_indices, :, :]  # (B, len(spatial), H, W)
        conv_feat = self.conv_head(spatial_in)         # (B, conv_out * H * W)

        # 2) Extract raw score scalars directly
        scores = [x[:, idx, 0, 0].unsqueeze(1) for idx in self.score_indices]
        scores = torch.cat(scores, dim=1)              # (B, n_scores)

        # 3) Concatenate and predict
        combined = torch.cat([conv_feat, scores], dim=1)  # (B, total_dim)
        return self.mlp(combined).squeeze(-1)          
    
class StateEvaluatorCNN(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[512, 256, 128]):
        super(StateEvaluatorCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        conv_output_dim = 128 * 6 * 6

        self.mlp = nn.Sequential(
            nn.Linear(conv_output_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dims[2], 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        output = self.mlp(x)
        return output.squeeze(-1)