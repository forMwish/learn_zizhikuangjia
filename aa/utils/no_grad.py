from ..core.config import Config


class no_grad:
    def __init__(self):
        pass

    def __enter__(self):
        Config.enable_backprop = False
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            raise exc_type (f"{exc_type} {exc_value} {traceback}")
        Config.enable_backprop = True
