# 윈디, 샤비
class CustomTrainer(Trainer):
        def create_optimizer_and_scheduler(self, num_training_steps: int):
            self.optimizer = AdamW([
                {'params': self.model.model.parameters(), 'lr': 4e-5, 'weight_decay': 0.005},
                {'params': self.model.bi_lstm.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                {'params': self.model.linear.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
            ])

            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps
            )


# 학습률 스케줄러 ReduceLROnPlateau
class CustomTrainer(Trainer):
        def create_optimizer_and_scheduler(self, num_training_steps: int):
            self.optimizer = AdamW([
                {'params': self.model.parameters(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay}
            ])

        # Initialize the ReduceLROnPlateau scheduler
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=5, verbose=True)

        def on_epoch_end(self, args, state, control, logs=None, **kwargs):
            # Assume f1_score is the metric you want to use to reduce the learning rate
            f1_score = logs.get('eval_f1', None)
            if f1_score is not None:
                # Update the learning rate
                self.lr_scheduler.step(f1_score)
              

# 꼬링크, test2, 수댕이, 갸라도스
    class CustomTrainer(Trainer):
        def create_optimizer_and_scheduler(self, num_training_steps: int):
            self.optimizer = AdamW([
            {'params': self.model.model.parameters(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay},  # 1e-5, 5e-4
            {'params': self.model.bi_lstm.parameters(), 'lr': 1e-3, 'weight_decay': 5e-3},
            {'params': self.model.linear.parameters(), 'lr': 1e-3, 'weight_decay': 5e-3},
        ])

            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.args.warmup_steps,  # 1700
                num_training_steps=num_training_steps  # 17768
            )
