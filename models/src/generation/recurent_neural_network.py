import gc
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
class LSTMLanguageModelS(nn.Module):
    def __init__(self, embedding_model, decoder_vocab_size, embedding_dim=384,
                 decoder_embed_dim=256, hidden_dim=512, num_layers=2, dropout_p=0.2):
        super().__init__()

        # Title embedding model (frozen)
        self.title_embedding_transformer = embedding_model
        for param in self.title_embedding_transformer.parameters():
            param.requires_grad = False

        # Projections
        self.title_to_hidden = nn.Linear(embedding_dim, hidden_dim * num_layers)
        self.title_to_cell = nn.Linear(embedding_dim, hidden_dim * num_layers)
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, decoder_embed_dim)
        self.lstm_input_proj = nn.Linear(decoder_embed_dim + embedding_dim, decoder_embed_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.decoder_vocab_size = decoder_vocab_size
        self.embedding_dim = embedding_dim
        self.decoder_embed_dim = decoder_embed_dim


        # LSTM
        self.lstm = nn.LSTM(
            input_size=decoder_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0
        )

        # Output
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim * 2, decoder_vocab_size)
        )

    def _get_title_embeddings(self, title_embeddings):
        """Process pre-computed title embeddings"""
        # No need to re-encode since we're directly passing tensor embeddings
        # Just make sure they're on the right device
        device = next(self.parameters()).device
        if title_embeddings.device != device:
            title_embeddings = title_embeddings.to(device)
        return title_embeddings

    def forward(self, title_embeddings, decoder_inputs, lengths=None):
        batch_size = title_embeddings.size(0)

        # 1. Initialize LSTM states (multi-layer fix)
        h_0 = self.title_to_hidden(title_embeddings)
        c_0 = self.title_to_cell(title_embeddings)
        h_0 = h_0.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c_0 = c_0.view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        # 2. Enhanced title conditioning
        embedded = self.decoder_embedding(decoder_inputs)
        embedded = self.dropout(embedded)
        title_expanded = title_embeddings.unsqueeze(1).expand(-1, embedded.size(1), -1)
        combined = torch.cat([embedded, title_expanded], dim=-1)
        lstm_input = self.lstm_input_proj(combined)

        # 3. Packed sequences (unchanged logic)
        if lengths is not None:
            lstm_input = torch.nn.utils.rnn.pack_padded_sequence(
                lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=True
            )

        # 4. LSTM forward
        lstm_out, _ = self.lstm(lstm_input, (h_0, c_0))

        if lengths is not None:
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # 5. Improved output projection
        hidden_activated = torch.relu(lstm_out)
        output_logits = self.output_projection(hidden_activated)

        return output_logits

    def predict_next_token(self, title_embedding, current_token, hidden=None):
        """Generate next token with proper conditioning"""
        # Ensure title_embedding has batch dimension
        if title_embedding.dim() == 1:
            title_embedding = title_embedding.unsqueeze(0)  # [1, embedding_dim]

        batch_size = title_embedding.size(0)

        # Initialize hidden state if needed
        if hidden is None:
            h_0 = self.title_to_hidden(title_embedding)
            c_0 = self.title_to_cell(title_embedding)

            # Reshape to [num_layers, batch_size, hidden_dim]
            h_0 = h_0.view(batch_size, self.num_layers, self.hidden_dim)
            h_0 = h_0.permute(1, 0, 2).contiguous()

            c_0 = c_0.view(batch_size, self.num_layers, self.hidden_dim)
            c_0 = c_0.permute(1, 0, 2).contiguous()

            hidden = (h_0, c_0)

        # Ensure current_token is properly shaped
        if current_token.dim() == 1:
            current_token = current_token.unsqueeze(0)

        # Embed token
        embedded = self.decoder_embedding(current_token)  # [batch_size, 1, embed_dim]

        # *** Apply the same title conditioning as in forward() ***
        title_expanded = title_embedding.unsqueeze(1)  # [batch_size, 1, embed_dim]
        combined = torch.cat([embedded, title_expanded], dim=-1)
        lstm_input = self.lstm_input_proj(combined)

        # Forward through LSTM
        lstm_out, new_hidden = self.lstm(lstm_input, hidden)

        # Project to vocabulary with the full pipeline
        hidden_activated = torch.relu(lstm_out)
        output = self.output_projection(hidden_activated.squeeze(1))

        return output, new_hidden

    def train_lstm_model(self, model, dataloader, val_dataloader, optimizer, criterion, num_epochs=5, clip_grad=1.0,save_path="best_lstm_model.pt"):
        """Train the LSTM model using the optimized dataloader"""
        device = next(model.parameters()).device
        losses = []
        best_loss = float('inf')
        total_steps = num_epochs * len(dataloader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.001,
            total_steps=total_steps,
            pct_start=0.3,  # Spend 30% of training warming up
            anneal_strategy='cos'
        )

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            batch_count = 0

            # Progress tracking
            progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch in dataloader:
                batch_count += 1

                # Move batch data to device
                title_embeddings = batch['title_embeddings'].to(device)
                desc_inputs = batch['desc_inputs'].to(device)
                desc_targets = batch['desc_targets'].to(device)
                lengths = batch['lengths'].to(device)

                # Clear previous gradients
                optimizer.zero_grad()

                try:
                    # Forward pass
                    # Update model.forward() to accept the length parameter
                    outputs = model(title_embeddings, desc_inputs, lengths)

                    # Flatten predictions and targets for loss calculation
                    # outputs shape: [batch_size, seq_len, vocab_size]
                    flattened_outputs = outputs.reshape(-1, outputs.size(-1))
                    flattened_targets = desc_targets.reshape(-1)

                    # Calculate loss (ignore padding)

                    loss = criterion(flattened_outputs, flattened_targets)

                    # Backward pass
                    loss.backward()

                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                    # Update weights
                    optimizer.step()

                    # Update scheduler
                    scheduler.step()

                    # Track statistics
                    epoch_loss += loss.item()

                    # Update progress bar
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                    progress_bar.update(1)

                    # Periodically clear cache during training (every 50 batches)
                    if batch_count % 50 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM error in batch. Skipping batch and clearing cache.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Optionally reduce batch size dynamically here
                    else:
                        raise e

            # Close progress bar
            progress_bar.close()

            # Calculate average loss for the epoch
            avg_loss = epoch_loss / batch_count
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs} complete, Avg Loss: {avg_loss:.4f}")
            if val_dataloader:
                val_loss = self.validate_model(model, val_dataloader, criterion, device)
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Save model if validation loss improved
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                        'loss': best_loss
                    }, save_path)
                    print(f"Model saved at epoch {epoch+1} with validation loss: {best_loss:.4f}")
                else:
                    # Without validation data, use training loss
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                            'loss': best_loss
                        }, save_path)
                        print(f"Model saved at epoch {epoch+1} with training loss: {best_loss:.4f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return losses

    def validate_model(self,model, dataloader, criterion, device):
        """Evaluate model on validation data"""
        model.eval()
        total_loss = 0
        batch_count = 0

        with torch.no_grad():
            for batch in dataloader:
                title_embeddings = batch['title_embeddings'].to(device)
                desc_inputs = batch['desc_inputs'].to(device)
                desc_targets = batch['desc_targets'].to(device)
                lengths = batch['lengths'].to(device)

                # Forward pass
                outputs = model(title_embeddings, desc_inputs, lengths)

                # Calculate loss
                flattened_outputs = outputs.reshape(-1, outputs.size(-1))
                flattened_targets = desc_targets.reshape(-1)
                loss = criterion(flattened_outputs, flattened_targets)

                total_loss += loss.item()
                batch_count += 1

        # Average validation loss
        return total_loss / max(1, batch_count)