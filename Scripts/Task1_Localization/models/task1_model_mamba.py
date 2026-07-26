# Definizione dell'architettura Mamba per la classificazione sequenziale
import torch
import torch.nn as nn
try:
    from mambapy.mamba import Mamba, MambaConfig
    MAMBAPY_AVAILABLE = True
except ImportError:
    MAMBAPY_AVAILABLE = False

class MambaBlock(nn.Module):

    def __init__(self, d_model: int, d_state: int=16, d_conv: int=4, expand: int=2):
        super().__init__()
        if not MAMBAPY_AVAILABLE:
            raise ImportError('mambapy non trovato. Installare con: pip install --user mambapy\nAssicurarsi di essere nel container latest.sif.')
        self.norm = nn.LayerNorm(d_model)
        config = MambaConfig(d_model=d_model, n_layers=1, d_state=d_state, d_conv=d_conv, expand_factor=expand)
        full_mamba = Mamba(config)
        self.mamba = full_mamba.layers[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))

class MambaRoomLocalizer(nn.Module):

    def __init__(self, input_dim: int=384, d_model: int=256, num_layers: int=4, num_classes: int=22, d_state: int=16, d_conv: int=4, expand: int=2, dropout: float=0.1):
        super().__init__()
        self.input_projection = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        self.dropout = nn.Dropout(dropout)
        self.mamba_layers = nn.ModuleList([MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        n_params = sum((p.numel() for p in self.parameters() if p.requires_grad))
        print(f'[MambaRoomLocalizer]\n  input_dim={input_dim} → d_model={d_model}\n  num_layers={num_layers} | num_classes={num_classes}\n  d_state={d_state} | d_conv={d_conv} | expand={expand} | dropout={dropout}\n  Parametri trainabili: {n_params:,}')

    def forward(self, x: torch.Tensor, lengths: torch.Tensor=None) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.dropout(x)
        for layer in self.mamba_layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.classifier(x)
        return logits

    def count_parameters(self) -> int:
        return sum((p.numel() for p in self.parameters() if p.requires_grad))
if __name__ == '__main__':
    import torch
    if not MAMBAPY_AVAILABLE:
        print('ERRORE: mambapy non installata. Usare: pip install --user mambapy')
        exit(1)
    print('=== Test MambaRoomLocalizer ===\n')
    B, T, D = (2, 100, 384)
    num_classes = 22
    model = MambaRoomLocalizer(input_dim=D, d_model=128, num_layers=2, num_classes=num_classes, d_state=16)
    x = torch.randn(B, T, D)
    lengths = torch.tensor([T, T])
    logits = model(x, lengths)
    print(f'\nInput:   {x.shape}')
    print(f'Output:  {logits.shape}  (atteso: [{B}, {T}, {num_classes}])')
    assert logits.shape == (B, T, num_classes), 'Shape output errato!'
    print(f'Parametri: {model.count_parameters():,}')
    print('\n✓ MambaRoomLocalizer funzionante.')