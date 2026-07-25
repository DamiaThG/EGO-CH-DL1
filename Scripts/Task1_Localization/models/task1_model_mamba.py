"""
Task 1 — Room-based Localization: Modello Mamba SSM

Architettura:
    Input [B, T, 384]  (feature DINOv2 pre-estratte)
        ↓
    Linear Projection → [B, T, d_model]
        ↓
    N × MambaBlock (pre-norm + SSM + skip connection)
        ↓
    LayerNorm finale
        ↓
    Linear Classifier → [B, T, num_classes]
        ↓
    Output [B, T, num_classes]  (logit per ogni frame)

Backend: mambapy (pip install --user mambapy in latest.sif)
         Nessun kernel CUDA custom richiesto — funziona su latest.sif.
"""
import torch
import torch.nn as nn

# mambapy deve essere installata nel container via: pip install --user mambapy
try:
    from mambapy.mamba import Mamba, MambaConfig
    MAMBAPY_AVAILABLE = True
except ImportError:
    MAMBAPY_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Blocco Mamba singolo
# ─────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """
    Singolo blocco Mamba con pre-norm e skip connection.
    Schema: x = x + SSM(LayerNorm(x))

    Args:
        d_model:  dimensione dell'embedding interno
        d_state:  dimensione dello stato nascosto SSM (default 16)
        d_conv:   dimensione della convoluzione 1D interna (default 4)
        expand:   fattore di espansione della proiezione interna (default 2)
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()

        if not MAMBAPY_AVAILABLE:
            raise ImportError(
                "mambapy non trovato. Installare con: pip install --user mambapy\n"
                "Assicurarsi di essere nel container latest.sif."
            )

        self.norm = nn.LayerNorm(d_model)

        config = MambaConfig(
            d_model=d_model,
            n_layers=1,       # usiamo n_layers=1 e prendiamo solo il primo layer
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand,
        )
        # Istanzia il modello Mamba completo e estrae il singolo layer SSM
        full_mamba = Mamba(config)
        self.mamba = full_mamba.layers[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        # Pre-norm + SSM + skip connection
        return x + self.mamba(self.norm(x))


# ─────────────────────────────────────────────────────────────────────────────
# Modello principale
# ─────────────────────────────────────────────────────────────────────────────

class MambaRoomLocalizer(nn.Module):
    """
    Classificatore per Room-based Localization basato su Mamba SSM.
    Opera su sequenze temporali di feature DINOv2.

    Input:  [B, T, input_dim]    — feature DINOv2 (input_dim=384)
    Output: [B, T, num_classes]  — logit per ogni frame

    Complessità computazionale: O(T) in memoria e computazione durante
    l'inferenza (vantaggio chiave rispetto a Transformer O(T²)).

    Args:
        input_dim:   dimensione delle feature di input (384 per DINOv2 ViT-S/14)
        d_model:     dimensione interna del modello Mamba
        num_layers:  numero di blocchi Mamba in cascata
        num_classes: numero di stanze (22 per Bellomo, 4 per Monastero)
        d_state:     dimensione dello stato SSM
        d_conv:      dimensione della conv interna Mamba
        expand:      fattore di espansione interno
        dropout:     dropout applicato dopo la proiezione iniziale
    """

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

        # Proiezione lineare: da spazio DINOv2 (384) a spazio Mamba (d_model)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.dropout = nn.Dropout(dropout)

        # Stack di blocchi Mamba
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state,
                       d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

        # Normalizzazione finale
        self.final_norm = nn.LayerNorm(d_model)

        # Testa di classificazione per-frame
        self.classifier = nn.Linear(d_model, num_classes)

        # Log architettura
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[MambaRoomLocalizer]\n"
            f"  input_dim={input_dim} → d_model={d_model}\n"
            f"  num_layers={num_layers} | num_classes={num_classes}\n"
            f"  d_state={d_state} | d_conv={d_conv} | expand={expand} | dropout={dropout}\n"
            f"  Parametri trainabili: {n_params:,}"
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:       Tensor[B, T, input_dim] — sequenze di feature (con padding)
            lengths: LongTensor[B]           — lunghezze originali (non usate
                     nel forward di Mamba, ma accettate per compatibilità
                     con l'interfaccia dei modelli baseline)

        Returns:
            logits: Tensor[B, T, num_classes]
        """
        # 1. Proiezione + dropout
        x = self.input_projection(x)   # [B, T, d_model]
        x = self.dropout(x)

        # 2. Stack Mamba
        for layer in self.mamba_layers:
            x = layer(x)               # [B, T, d_model]

        # 3. Normalizzazione finale
        x = self.final_norm(x)

        # 4. Classificazione per-frame
        logits = self.classifier(x)    # [B, T, num_classes]
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Test rapido
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    if not MAMBAPY_AVAILABLE:
        print("ERRORE: mambapy non installata. Usare: pip install --user mambapy")
        exit(1)

    print("=== Test MambaRoomLocalizer ===\n")

    B, T, D = 2, 100, 384  # batch=2, sequenza=100 frame, DINOv2=384
    num_classes = 22

    model = MambaRoomLocalizer(
        input_dim=D, d_model=128, num_layers=2,
        num_classes=num_classes, d_state=16
    )

    x = torch.randn(B, T, D)
    lengths = torch.tensor([T, T])

    logits = model(x, lengths)
    print(f"\nInput:   {x.shape}")
    print(f"Output:  {logits.shape}  (atteso: [{B}, {T}, {num_classes}])")
    assert logits.shape == (B, T, num_classes), "Shape output errato!"

    print(f"Parametri: {model.count_parameters():,}")
    print("\n✓ MambaRoomLocalizer funzionante.")
