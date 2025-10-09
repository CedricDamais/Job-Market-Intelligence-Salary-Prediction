```mermaid
graph LR
    subgraph "Initialisation"
        T[Titre] --> TE["Embedding Titre"]
        TE --> H0["h₀ = Linear(TE)"]
        TE --> C0["c₀ = Linear(TE)"]
    end

    subgraph "LSTM Cell"
        W_t["Embedding Token t"] --> CONCAT["Concaténation"]
        TE --> CONCAT
        CONCAT --> LSTM["LSTM"]
        H0 --> LSTM
        C0 --> LSTM
        LSTM --> H_t["h_t"]
        LSTM --> C_t["c_t"]
        H_t --> P_t["Prédiction t"]
        
        %% Boucle de récurrence
        H_t -.-> |"t+1"| LSTM
        C_t -.-> |"t+1"| LSTM
    end

    style T fill:#D90429,color:white
    style TE fill:#D90429,color:white
    style W_t fill:#0077B6,color:white
    style CONCAT fill:#9B5DE5,color:white
    style LSTM fill:#2A9D8F,color:white
    style P_t fill:#E9C46A,color:black
```

## Architecture du Modèle avec Récurrence

1. **Initialisation**
   - Le titre est transformé en embedding
   - Les états initiaux h₀ et c₀ sont calculés à partir de l'embedding du titre

2. **LSTM Cell Récurrente**
   - À chaque pas de temps t :
     * L'embedding du token courant est concaténé avec l'embedding du titre
     * Le LSTM utilise les états précédents (h_t-1, c_t-1)
     * Génère les nouveaux états (h_t, c_t)
     * Produit la prédiction pour le token t
   - Les états sont réutilisés au pas de temps suivant (t+1)
   - La mémoire à long terme est maintenue via c_t 