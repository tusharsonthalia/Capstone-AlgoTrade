import time
import sys


class ProgressBar:
    "Class to display progress bar in the command line."

    def __init__(self, file=sys.stdout, style='█-', bar_length=30):
        """
        Parameters:
        ---------
        file:       file-like object, default sys.stdout
                    The output location to display the progress bar.

        style:      str, default '█-'
                    The style of the progress bar. The first character is
                    for the progress done and the second character is for 
                    progress to be done.

        bar_length: int, default 30
                    The length of the progress bar.
        """
        # raising error if the style is inappropriate
        if len(style) != 2:
            raise ValueError(f"Style is inappropriate ('{style}'). "
                             "Please use something similar to '#-'")
        self.file = file
        self.s1, self.s2 = style
        self.bar_length = bar_length

    def _write_progress(self, status=""):
        """
        Private method to update the console display.

        Parameters:
        ---------
        status: str, default "", optional
                    The user message that should be updated on the progress bar.
        """
        done = 0
        elapsed = self.current_time - self.start_time
        eta = "-"
        iterations = '0 it/s'
        percentage = '0%'
        if self.steps_done:
            done = int((self.steps_done/self.steps_to_do) * self.bar_length)
            iterations = f"{round(self.steps_done / elapsed, 2)} it/s"
            eta = ((elapsed * self.steps_to_do)/self.steps_done) - elapsed
            eta = f"{time.strftime('%H:%M:%S', time.gmtime(round(eta, 2)))}"
            percentage = f"{round((self.steps_done / self.steps_to_do) * 100, 2)}%"

        elapsed = f"{time.strftime('%H:%M:%S', time.gmtime(round(elapsed, 2)))}"
        to_do = self.bar_length - done
        progress_bar = f"[{self.s1*done}{self.s2*to_do}]"
        progress_steps = f"{self.steps_done}/{self.steps_to_do}"
        if status:
            status = f' {status} |'
        progress = f"\rProgress: {progress_bar} |{status} {progress_steps} | {percentage} | {elapsed}<{eta} | {iterations}"
        self.file.write(progress)
        self.file.flush()

    def start_progress(self, steps_to_do):
        """
        Method to start the progress bar.

        Parameters:
        ---------
        steps_to_do:    int 
                        The number of steps that are there in the process.
        """
        self.start_time = self.current_time = time.time()
        self.steps_done = 0
        self.steps_to_do = steps_to_do
        self._write_progress()

    def update_progress(self, status=""):
        """
        Method to update the progress bar.

        Parameters:
        ---------
        status: str, default "", optional
                    The user message that should be updated on the progress bar.
        """
        self.current_time = time.time()
        self.steps_done += 1
        self._write_progress(status)


if __name__ == '__main__':
    progress = ProgressBar()
    progress.start_progress(500)
    for _ in range(500):
        time.sleep(0.01)
        progress.update_progress()
