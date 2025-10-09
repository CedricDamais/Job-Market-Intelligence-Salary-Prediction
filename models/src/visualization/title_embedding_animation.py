import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle, Circle, FancyArrow
import matplotlib.colors as mcolors

plt.style.use('seaborn-v0_8-whitegrid')

class TitleEmbeddingAnimation:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.ax.set_xlim(0, 13.5)
        self.ax.set_ylim(0, 8)
        self.ax.axis('off')
        
        # Palette de couleurs améliorée
        self.colors = {
            'title': '#D90429',          # Rouge fort
            'token_current': '#0077B6',  # Bleu vif (entrée actuelle)
            'token_processed': '#CAF0F8', # Bleu ciel clair (entrées traitées)
            'lstm_base': '#2A9D8F',      # Bleu-vert (LSTM)
            'lstm_active': '#FCA311',    # Orange vif (LSTM en traitement)
            'prediction_bg': '#E9C46A',  # Jaune sable (fond prédiction)
            'prediction_text': '#2B2D42', # Texte sombre pour prédiction
            'generated_token': '#2A9D8F', # Bleu-vert (token généré)
            'arrow': '#333333',          # Gris foncé pour flèches
            'text_light': '#FFFFFF',     # Blanc pour texte sur fond sombre
            'text_dark': '#2B2D42',      # Gris foncé pour texte général
            'legend_bg': '#EDF2F4',      # Gris très clair pour légende
            'concat': '#9B5DE5'          # Violet pour la concaténation
        }
        
        # Éléments actifs à mettre à jour
        self.active_elements = []
        
        # Configuration des blocs fixes
        self.setup_fixed_elements()
        
        # Données pour l'animation
        self.tokens = ["[Début]", "Nouveau", "Smartphone", "Haute", "Performance", "[Fin]"]
        self.num_tokens_to_process = len(self.tokens) - 1
        self.stages_per_token = 7  # Augmenté pour inclure l'étape de concaténation
        self.total_frames = self.num_tokens_to_process * self.stages_per_token + self.stages_per_token
        
        # Positions des tokens
        self.y_input_token_row = 4.8
        self.y_generated_token_row = 1.3
        self.plot_width_for_tokens = 9.0
        self.start_x_for_tokens = 1.0
        self.input_token_x_positions = np.linspace(
            self.start_x_for_tokens, 
            self.start_x_for_tokens + self.plot_width_for_tokens, 
            self.num_tokens_to_process
        )
        self.generated_token_x_positions = np.linspace(
            self.start_x_for_tokens, 
            self.start_x_for_tokens + self.plot_width_for_tokens, 
            self.num_tokens_to_process
        )

    def setup_fixed_elements(self):
        # Bloc d'embedding du titre
        title_rect_coords = (0.5, 6.5)
        title_rect_dims = (2.5, 1)
        self.ax.add_patch(Rectangle(title_rect_coords, *title_rect_dims, 
                                  fill=True, color=self.colors['title'], 
                                  ec='black', zorder=1))
        self.ax.text(title_rect_coords[0] + title_rect_dims[0]/2, 
                    title_rect_coords[1] + title_rect_dims[1]/2,
                    "TITRE\nEMBEDDING", ha='center', va='center', 
                    color=self.colors['text_light'], fontsize=9, weight='bold')

        # Bloc de concaténation
        concat_rect_coords = (3.5, 6.5)
        concat_rect_dims = (1, 1)
        self.ax.add_patch(Rectangle(concat_rect_coords, *concat_rect_dims, 
                                  fill=True, color=self.colors['concat'], 
                                  ec='black', zorder=1))
        self.ax.text(concat_rect_coords[0] + concat_rect_dims[0]/2, 
                    concat_rect_coords[1] + concat_rect_dims[1]/2,
                    "⊕", ha='center', va='center', 
                    color=self.colors['text_light'], fontsize=14, weight='bold')

        # Bloc LSTM
        lstm_rect_coords = (5.5, 6.5)
        lstm_rect_dims = (3, 1)
        self.lstm_main_rect = Rectangle(lstm_rect_coords, *lstm_rect_dims, 
                                      fill=True, color=self.colors['lstm_base'], 
                                      ec='black', zorder=1)
        self.ax.add_patch(self.lstm_main_rect)
        self.ax.text(lstm_rect_coords[0] + lstm_rect_dims[0]/2, 
                    lstm_rect_coords[1] + lstm_rect_dims[1]/2,
                    "LSTM", ha='center', va='center', 
                    color=self.colors['text_light'], fontsize=11, weight='bold')

        # Bloc de prédiction
        pred_rect_coords = (9.5, 6.5)
        pred_rect_dims = (2.5, 1)
        self.ax.add_patch(Rectangle(pred_rect_coords, *pred_rect_dims, 
                                  fill=True, color=self.colors['prediction_bg'], 
                                  ec='black', zorder=1))
        self.ax.text(pred_rect_coords[0] + pred_rect_dims[0]/2, 
                    pred_rect_coords[1] + pred_rect_dims[1]/2,
                    "PRÉDICTION", ha='center', va='center', 
                    color=self.colors['prediction_text'], fontsize=9, weight='bold')

        # Labels pour les rangées de tokens
        self.ax.text(0.2, 5.0, "Entrée Actuelle (Tᵢ):", 
                    ha='left', va='center', fontsize=10, 
                    color=self.colors['text_dark'], weight='semibold')
        self.ax.text(0.2, 1.5, "Séquence Générée (Pᵢ → Tᵢ₊₁):", 
                    ha='left', va='center', fontsize=10, 
                    color=self.colors['text_dark'], weight='semibold')

        # Légende déplacée en bas
        self.setup_legend()

    def setup_legend(self):
        legend_items = [
            (self.colors['title'], "Embedding Titre"),
            (self.colors['token_current'], "Token d'Entrée (Tᵢ)"),
            (self.colors['concat'], "Concaténation"),
            (self.colors['lstm_active'], "LSTM en Traitement"),
            (self.colors['prediction_bg'], "Zone de Prédiction (Pᵢ)"),
            (self.colors['generated_token'], "Token Prédit/Généré")
        ]
        
        # Légende centrée en bas
        legend_box_coords = (4.5, 0.5)
        legend_box_dims = (3, 0.8)
        
        self.ax.add_patch(Rectangle(legend_box_coords, *legend_box_dims, 
                                  fill=True, color=self.colors['legend_bg'], 
                                  ec='gray', alpha=0.9))
        
        # Disposition horizontale des éléments de la légende
        for i, (color, label) in enumerate(legend_items):
            x_pos = legend_box_coords[0] + 0.3 + (i * 1.8)
            y_pos = legend_box_coords[1] + legend_box_dims[1]/2
            self.ax.add_patch(Rectangle((x_pos - 0.15, y_pos - 0.125), 
                                      0.3, 0.25, color=color, ec='black'))
            self.ax.text(x_pos + 0.2, y_pos, label, 
                        ha='left', va='center', fontsize=8, 
                        color=self.colors['text_dark'])

    def clear_dynamic_elements(self):
        for el in self.active_elements:
            el.remove()
        self.active_elements.clear()
        self.lstm_main_rect.set_facecolor(self.colors['lstm_base'])

    def update(self, frame_num):
        self.clear_dynamic_elements()
        
        current_token_idx = frame_num // self.stages_per_token
        stage = frame_num % self.stages_per_token

        # Afficher les tokens d'entrée traités
        for i in range(current_token_idx):
            if i < self.num_tokens_to_process:
                text_obj = self.ax.text(
                    self.input_token_x_positions[i], 
                    self.y_input_token_row, 
                    self.tokens[i], 
                    ha='center', va='center', 
                    fontsize=8, 
                    color=self.colors['text_dark'],
                    bbox=dict(facecolor=self.colors['token_processed'], 
                            alpha=0.8, 
                            boxstyle='round,pad=0.4', 
                            ec='grey')
                )
                self.active_elements.append(text_obj)

        # Afficher les tokens générés
        limit_generated_display = current_token_idx if stage < 6 else current_token_idx + 1
        for i in range(min(limit_generated_display, self.num_tokens_to_process)):
            gen_token_obj = self.ax.text(
                self.generated_token_x_positions[i], 
                self.y_generated_token_row, 
                self.tokens[i+1], 
                ha='center', va='center', 
                fontsize=9, 
                color=self.colors['text_light'],
                bbox=dict(facecolor=self.colors['generated_token'], 
                         boxstyle='round,pad=0.5', 
                         ec='black', 
                         lw=1)
            )
            self.active_elements.append(gen_token_obj)

        if current_token_idx >= self.num_tokens_to_process:
            final_text = self.ax.text(
                (5.5 + 3/2), 3.5, 
                "Génération Terminée!", 
                ha='center', va='center', 
                fontsize=14, 
                color='darkgreen', 
                weight='bold'
            )
            self.active_elements.append(final_text)
            return self.active_elements

        # Traitement du token courant
        current_input_token_text = self.tokens[current_token_idx]
        predicted_token_text = self.tokens[current_token_idx+1]
        current_input_x = self.input_token_x_positions[current_token_idx]

        # Étape 0: Mettre en évidence le token d'entrée courant
        input_token_obj = self.ax.text(
            current_input_x, 
            self.y_input_token_row, 
            current_input_token_text, 
            ha='center', va='center', 
            fontsize=9, 
            color=self.colors['text_light'], 
            weight='bold',
            bbox=dict(facecolor=self.colors['token_current'], 
                     boxstyle='round,pad=0.5', 
                     ec='black', 
                     lw=1.5), 
            zorder=10
        )
        self.active_elements.append(input_token_obj)

        # Étape 1: Flèche d'initialisation du LSTM par le titre
        if stage >= 1:
            arrow_title_lstm = FancyArrow(
                1.75, 
                7.0,
                3.75, 
                0,
                width=0.02, 
                head_width=0.15, 
                head_length=0.25, 
                color=self.colors['title'], 
                alpha=0.9, 
                zorder=5
            )
            self.active_elements.append(self.ax.add_patch(arrow_title_lstm))
            
            # Texte d'initialisation
            init_text = self.ax.text(
                3.0, 
                7.3, 
                "Initialisation des états", 
                ha='center', va='center', 
                fontsize=8, 
                color=self.colors['title'], 
                weight='bold'
            )
            self.active_elements.append(init_text)

        # Étape 2: Flèche Tᵢ -> Concaténation
        if stage >= 2:
            arrow_ti_concat = FancyArrow(
                current_input_x, 
                self.y_input_token_row - 0.3,
                3.5 + 0.5 - current_input_x,
                6.5 + 0.5 - (self.y_input_token_row - 0.3),
                width=0.02, 
                head_width=0.15, 
                head_length=0.25, 
                color=self.colors['token_current'], 
                alpha=0.9, 
                zorder=5
            )
            self.active_elements.append(self.ax.add_patch(arrow_ti_concat))

        # Étape 3: Flèche Titre -> Concaténation
        if stage >= 3:
            arrow_title_concat = FancyArrow(
                1.75, 
                7.0,
                1.75, 
                0,
                width=0.02, 
                head_width=0.15, 
                head_length=0.25, 
                color=self.colors['title'], 
                alpha=0.9, 
                zorder=5
            )
            self.active_elements.append(self.ax.add_patch(arrow_title_concat))

        # Étape 4: Flèche Concaténation -> LSTM
        if stage >= 4:
            arrow_concat_lstm = FancyArrow(
                4.5, 
                7.0,
                1.0, 
                0,
                width=0.02, 
                head_width=0.15, 
                head_length=0.25, 
                color=self.colors['concat'], 
                alpha=0.9, 
                zorder=5
            )
            self.active_elements.append(self.ax.add_patch(arrow_concat_lstm))

        # Étape 5: LSTM en traitement
        if stage >= 5:
            self.lstm_main_rect.set_facecolor(self.colors['lstm_active'])
            status_obj = self.ax.text(
                5.5 + 1.5, 
                6.5 - 0.4, 
                "Traitement...",
                ha='center', va='center', 
                fontsize=8, 
                color=self.colors['text_dark'], 
                weight='semibold'
            )
            self.active_elements.append(status_obj)

        # Étape 6: Flèche LSTM -> Prédiction
        if stage >= 6:
            if stage == 6:
                self.lstm_main_rect.set_facecolor(self.colors['lstm_base'])
            arrow_lstm_pred = FancyArrow(
                8.5, 
                7.0,
                1.0, 
                0,
                width=0.02, 
                head_width=0.15, 
                head_length=0.25, 
                color=self.colors['lstm_base'], 
                alpha=0.9, 
                zorder=5
            )
            self.active_elements.append(self.ax.add_patch(arrow_lstm_pred))

            # Afficher le token prédit
            pred_text_obj = self.ax.text(
                10.75, 
                7.0,
                predicted_token_text, 
                ha='center', va='center', 
                fontsize=9, 
                color=self.colors['prediction_text'], 
                weight='bold',
                bbox=dict(facecolor='white', 
                         edgecolor=self.colors['prediction_text'], 
                         boxstyle='round,pad=0.5', 
                         lw=1.5), 
                zorder=10
            )
            self.active_elements.append(pred_text_obj)

        return self.active_elements

    def animate(self):
        anim = FuncAnimation(
            self.fig, 
            self.update, 
            frames=self.total_frames, 
            interval=600,
            blit=False,
            repeat=False
        )
        
        try:
            output_filename = 'title_conditioning_animation.gif'
            anim.save(
                output_filename,
                writer='pillow',
                fps=max(1, 1000//600),
                dpi=120,
                savefig_kwargs={'facecolor': self.fig.get_facecolor()}
            )
            print(f"Animation sauvegardée: {output_filename}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'animation: {e}")
            print("Vérifiez que Pillow est installé et que les dépendances GIF sont présentes.")
        
        plt.close(self.fig)

if __name__ == "__main__":
    animation = TitleEmbeddingAnimation()
    animation.animate() 