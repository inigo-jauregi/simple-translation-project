import torch
import pytorch_lightning as pl
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
import sacrebleu

class NLLBTranslationModel(pl.LightningModule):
    def __init__(
            self,
            model_name="facebook/nllb-200-distilled-600M",
            src_lng=None,
            tgt_lng=None,
            learning_rate=5e-5,
            warmup_steps=500,
            max_steps=10000
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.src_lng = src_lng
        self.tgt_lng = tgt_lng

        # Store validation outputs
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True)

        # Generate translations for BLEU calculation
        self.tokenizer.src_lang = self.tgt_lng
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lng),
            max_length=128,
            num_beams=4
        )

        # Decode predictions and references
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Decode labels (replace -100 with pad_token_id for decoding)
        labels = batch['labels'].clone()
        labels[labels == -100] = self.tokenizer.pad_token_id
        references = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Store for epoch end calculation
        self.validation_step_outputs.append({
            'predictions': predictions,
            'references': references
        })

        return loss

    def on_validation_epoch_end(self):
        # Collect all predictions and references
        all_predictions = []
        all_references = []

        for output in self.validation_step_outputs:
            all_predictions.extend(output['predictions'])
            all_references.extend(output['references'])

        # Calculate BLEU score
        bleu = sacrebleu.corpus_bleu(all_predictions, [all_references], tokenize="ja-mecab")

        # Log BLEU score
        self.log('val_bleu', bleu.score, prog_bar=True)
        print(f"\nValidation BLEU: {bleu.score:.2f}")

        # Clear stored outputs
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.max_steps
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }