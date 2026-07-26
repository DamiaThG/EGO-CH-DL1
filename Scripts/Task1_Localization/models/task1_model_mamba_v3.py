"""
Task 1 — Room-based Localization: Modello Mamba SSM V3
Architettura con:
1. MLP Projection Layer (anziché singolo Linear)
2. Mamba Blocks
3. MS-TCN Head (Smoothing neurale tramite Convoluzioni Dilatate)
"""
import torch
import torch.nn as nn

try:
    from mambapy.mamba import Mamba, MambaConfig
    MAMBAPY_AVAILABLE = True
except ImportError:
    MAMBAPY_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# MS-TCN Head (Smoothing Neurale end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

class DilatedTCNBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        # Kernel 3 con padding uguale alla dilation mantiene la lunghezza inalterata
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        # GroupNorm(1) equivale a un LayerNorm sul canale spaziale, senza mescolare i sample del batch
        self.norm = nn.GroupNorm(1, channels)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: [B, C, T]
        return x + self.activation(self.norm(self.conv(x)))

class TCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=4):
        super().__init__()
        layers = []
        # Aumentiamo la dilation in modo esponenziale per allargare il campo recettivo
        for i in range(num_layers):
            layers.append(DilatedTCNBlock(in_channels, dilation=2**i))
        
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Mamba restituisce [B, T, C], la Conv1d vuole [B, C, T]
        x = x.transpose(1, 2)
        x = self.tcn(x)
        logits = self.classifier(x)
        # Riportiamo a [B, T, C]
        return logits.transpose(1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Mamba Block Originale
# ─────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        if not MAMBAPY_AVAILABLE:
            raise ImportError("mambapy non trovato.")
        self.norm = nn.LayerNorm(d_model)
        config = MambaConfig(
            d_model=d_model, n_layers=1, d_state=d_state,
            d_conv=d_conv, expand_factor=expand,
        )
        full_mamba = Mamba(config)
        self.mamba = full_mamba.layers[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


# ─────────────────────────────────────────────────────────────────────────────
# Modello principale V3
# ─────────────────────────────────────────────────────────────────────────────

class MambaRoomLocalizerV3(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        d_model: int = 256,
        num_layers: int = 4,
        num_classes: int = 22,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1. MLP Projection Layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.dropout = nn.Dropout(dropout)

        # 2. Mamba Blocks
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state,
                       d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # 3. MS-TCN Head (Smoothing)
        self.classifier = TCNHead(in_channels=d_model, out_channels=num_classes, num_layers=4)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[MambaRoomLocalizerV3]\n"
            f"  input_dim={input_dim} → d_model={d_model}\n"
            f"  num_layers={num_layers} | num_classes={num_classes}\n"
            f"  d_state={d_state} | d_conv={d_conv} | expand={expand} | dropout={dropout}\n"
            f"  Parametri trainabili: {n_params:,}"
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.dropout(x)

        for layer in self.mamba_layers:
            x = layer(x)

        x = self.final_norm(x)
        
        logits = self.classifier(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    B, T, D = 2, 100, 384
    model = MambaRoomLocalizerV3(input_dim=D, d_model=128, num_layers=2, num_classes=22)
    x = torch.randn(B, T, D)
    logits = model(x)
    assert logits.shape == (B, T, 22)
    print("V3 Model check passed!")
