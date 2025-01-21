class MNISTBoostingPipeline:
    """Orchestrates the complete boosting training pipeline including ensemble evaluation."""
    
    def __init__(self, iterations: int = 3, model_name: str = "deepseek/deepseek-chat"):
        self.iterations = iterations
        self.model_name = model_name
        self.ensemble = MNISTBooster(
            model_name=model_name,
            boosting_iterations=iterations
        )
        self.evaluator = MNISTEvaluator(model_name=model_name)
