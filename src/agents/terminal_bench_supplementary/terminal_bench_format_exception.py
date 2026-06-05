class terminal_bench_format_exception(Exception):
    """Exception raised for errors in the input format of TerminalBench."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
