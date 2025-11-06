import pytorch_lightning as pl

from model import NLLBTranslationModel
from dataloader import TranslationDataModule


def main():
    # Initialize model
    model = NLLBTranslationModel(
        model_name="facebook/nllb-200-distilled-600M",
        src_lng="eng_Latn",
        tgt_lng="zho_Hans",
        learning_rate=5e-5,
        warmup_steps=500,
        max_steps=10000
    )

    # Initialize data module
    data_module = TranslationDataModule(
        tokenizer=model.tokenizer,
        batch_size=8,
        max_samples=1000  # Adjust based on your resources
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_steps=10000,
        accelerator='auto',
        devices=1,
        max_epochs=1,
        precision='16-mixed',  # Use mixed precision for faster training
        gradient_clip_val=1.0,
        val_check_interval=1.0,
        log_every_n_steps=1,
        accumulate_grad_batches=4,  # Effective batch size = 8 * 4 = 32
    )

    # Train
    trainer.fit(model, data_module)

    # Save model
    model.model.save_pretrained("./nllb_finetuned")
    model.tokenizer.save_pretrained("./nllb_finetuned")
    print("Model saved to ./nllb_finetuned")


if __name__ == "__main__":
    main()
