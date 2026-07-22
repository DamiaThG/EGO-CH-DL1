"""
Task 1 — Room-based Localization: Modelli Baseline

Implementa MLP e LSTM con la stessa interfaccia del modello Mamba
(stesso signature del metodo forward) per permettere confronti puliti.

    forward(x: Tensor[B, T, input_dim], lengths: Tensor[B]) → Tensor[B, T, num_classes]

Modelli:
    - MLPRoomLocalizer:  classificazione frame-per-frame (nessun contesto temporale)
                         → lower bound per valutare il vantaggio di Mamba
    - LSTMRoomLocalizer: classificazione bidirezionale con memoria ricorrente
                         → riferimento classico pre-Transformer
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: MLP (frame-per-frame, no contesto temporale)
# ─────────────────────────────────────────────────────────────────────────────

class MLPRoomLocalizer(nn.Module):
    """
    MLP frame-per-frame. Non modella il contesto temporale.
    Serve come lower bound: se Mamba non supera questo, c'è un problema.

    Architettura:
        Linear(384, hidden) → ReLU → Dropout → Linear(hidden, hidden)
        → ReLU → Dropout → Linear(hidden, num_classes)

    Il forward è applicato indipendentemente a ogni frame.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_classes: int = 22,
        dropout: float = 0.1,
        num_hidden_layers: int = 2,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_classes))

        self.net = nn.Sequential(*layers)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[MLPRoomLocalizer]\n"
            f"  input_dim={input_dim} | hidden_dim={hidden_dim}\n"
            f"  num_hidden_layers={num_hidden_layers} | num_classes={num_classes}\n"
            f"  Parametri trainabili: {n_params:,}"
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       Tensor[B, T, input_dim]
            lengths: ignorato (MLP non usa contesto temporale)
        Returns:
            logits: Tensor[B, T, num_classes]
        """
        # Applica la rete a ogni frame indipendentemente
        return self.net(x)   # [B, T, num_classes]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: LSTM bidirezionale
# ─────────────────────────────────────────────────────────────────────────────

class LSTMRoomLocalizer(nn.Module):
    """
    LSTM bidirezionale con packed sequence per sequenze di lunghezza variabile.
    Modella le dipendenze temporali in entrambe le direzioni.

    Limitazione rispetto a Mamba: O(T) in computazione ma soffre di
    vanishing gradient su sequenze molto lunghe (~18k frame nei test).

    Architettura:
        Bi-LSTM(input_dim, hidden_dim, num_layers)
        → Linear(hidden_dim*2, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        num_classes: int = 22,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[LSTMRoomLocalizer]\n"
            f"  input_dim={input_dim} | hidden_dim={hidden_dim}\n"
            f"  num_layers={num_lstm_layers} | bidirectional={bidirectional}\n"
            f"  num_classes={num_classes}\n"
            f"  Parametri trainabili: {n_params:,}"
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       Tensor[B, T, input_dim]
            lengths: LongTensor[B] — lunghezze reali (per packed sequence)
        Returns:
            logits: Tensor[B, T, num_classes]
        """
        if lengths is not None:
            # Packed sequence: ignora il padding nella computazione LSTM
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output_packed, _ = self.lstm(packed)
            output, _ = pad_packed_sequence(output_packed, batch_first=True)
            # output: [B, T_max, hidden_dim*2]
        else:
            output, _ = self.lstm(x)

        return self.classifier(output)   # [B, T, num_classes]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Test rapido
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    print("=== Test Modelli Baseline ===\n")

    B, T, D = 2, 150, 384
    num_classes = 22
    lengths = torch.tensor([150, 120])  # seconda sequenza più corta

    x = torch.randn(B, T, D)

    # Test MLP
    mlp = MLPRoomLocalizer(input_dim=D, hidden_dim=256, num_classes=num_classes)
    out_mlp = mlp(x, lengths)
    print(f"\nMLP Output: {out_mlp.shape}  (atteso: [{B}, {T}, {num_classes}])")
    assert out_mlp.shape == (B, T, num_classes)

    # Test LSTM
    lstm = LSTMRoomLocalizer(input_dim=D, hidden_dim=256, num_classes=num_classes)
    out_lstm = lstm(x, lengths)
    print(f"LSTM Output: {out_lstm.shape}  (atteso: [{B}, {T}, {num_classes}])")
    assert out_lstm.shape == (B, T, num_classes)

    print("\n✓ Tutti i baseline funzionanti.")
