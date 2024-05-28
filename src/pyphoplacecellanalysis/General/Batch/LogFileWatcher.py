import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, PatternMatchingEventHandler, FileSystemEventHandler

class LogFileHandler(PatternMatchingEventHandler):
    """
    Requires `watchdog` library:
        pip install watchdog

    """
    patterns = ["*.log"]  # You can change the pattern to match the log files you need

    def process(self, event):
        """Prints the file path of log files that have been modified."""
        # The event's type is printed: CREATED, DELETED, or MODIFIED.
        # The event's src_path is the file path that you want to monitor.
        print("{} - {}.".format(event.src_path, event.event_type.upper()))

    def on_modified(self, event):
        self.process(event)

    def on_created(self, event):
        self.process(event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        return super().on_deleted(event)
    
    


if __name__ == "__main__":
    # path = r"K:\scratch\gen_scripts\run_kdiba_gor01_one_2006-6-08_14-26-15"  # Replace with the path to the directory you want to monitor
    path = r"K:\scratch\gen_scripts"  # Replace with the path to the directory you want to monitor
    event_handler = LogFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()