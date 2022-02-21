class UnsupportedError(Exception):
    def __init__(self, attr):
        super().__init__("Unsupported {}".format(attr))
        self.attr = attr

class UnsupportedTaskError(UnsupportedError):
    def __init__(self):
        super().__init__("task")

class UnsupportedActivationFuncError(UnsupportedError):
    def __init__(self):
        super().__init__("activation function")

class UnsupportedMetricError(UnsupportedError):
    def __init__(self):
        super().__init__("metric")

class UnsupportedFormatError(UnsupportedError):
    def __init__(self):
        super().__init__("format")

class UnsupportedSplitMethodError(UnsupportedError):
    def __init__(self):
        super().__init__("split method")

class UnsupportedModeError(UnsupportedError):
    def __init__(self):
        super().__init__("mode")

class UnsupportedModelError(UnsupportedError):
    def __init__(self):
        super().__init__("model")

class UnsupportedOptimizerError(UnsupportedError):
    def __init__(self):
        super().__init__("optimizer")

class UnsupportedPrivacyMechanismError(UnsupportedError):
    def __init__(self):
        super().__init__("privacy mechanism")
