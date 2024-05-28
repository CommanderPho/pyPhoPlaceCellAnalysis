import os
import glob
import time
import threading
import queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, PatternMatchingEventHandler, FileSystemEventHandler



def get_sorted_log_files(directory, recursive=True):
    """ 
    from pyphoplacecellanalysis.General.Batch.LogFileWatcher import get_sorted_log_files

    """
    if recursive:
        log_files = glob.glob(os.path.join(directory, '**', '*.log'), recursive=recursive)
    else:
        log_files = glob.glob(os.path.join(directory, '*.log'), recursive=recursive)

    log_files.sort(key=os.path.getmtime)
    return log_files




class LogFileHandler(PatternMatchingEventHandler):
    """
    Requires `watchdog` library:
        pip install watchdog

    """

    def __init__(self, patterns=None, log_queue=None, *args, **kwargs):
        if patterns is None:
            patterns = ["*.log"]  # You can change the pattern to match the log files you need
        super().__init__(patterns=patterns, *args, **kwargs)
        if log_queue is None:
            log_queue = queue.Queue()
        self.log_queue = log_queue
        self.file_offsets = {}


    def process(self, event):
        """Prints the file path of log files that have been modified."""
        # The event's type is printed: CREATED, DELETED, or MODIFIED.
        # The event's src_path is the file path that you want to monitor.
        print("{} - {}.".format(event.src_path, event.event_type.upper()))


    def on_change_update_file_offsets_and_print_new_lines(self, event):
        file_path = event.src_path
        new_lines = []

        # Read the file from the last offset position
        with open(file_path, 'r') as file:
            # If we haven't seen this file before, start at the current end of the file
            if file_path not in self.file_offsets:
                file.seek(0, 2)
            else:
                file.seek(self.file_offsets[file_path])

            # Read new lines and keep the offset
            for line in file:
                new_lines.append(line)

            # Update the stored offset position
            self.file_offsets[file_path] = file.tell()

        # Print the new lines found
        if new_lines:
            print(f"\tNew lines in {file_path}:")
            for line in new_lines:
                print(f"\t\t{line}", end='\n')


        contents: str = '\n'.join(new_lines)
        # Put file modification event into the queue
        self.log_queue.put((event.src_path, contents))



    def on_modified(self, event):
        self.process(event)
        self.on_change_update_file_offsets_and_print_new_lines(event=event)


    def on_created(self, event):
        self.process(event)
        self.on_change_update_file_offsets_and_print_new_lines(event=event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        return super().on_deleted(event)
    



# Start the watchdog observer
def start_monitoring(path, log_queue=None):
    event_handler = LogFileHandler(patterns=["*.log"], log_queue=log_queue)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    return observer


if __name__ == "__main__":
    # path = r"K:\scratch\gen_scripts\run_kdiba_gor01_one_2006-6-08_14-26-15"  # Replace with the path to the directory you want to monitor
    path = r"K:\scratch\gen_scripts"  # Replace with the path to the directory you want to monitor
    observer = start_monitoring(path=path)

    # event_handler = LogFileHandler()
    # observer = Observer()
    # observer.schedule(event_handler, path, recursive=True)
    # observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()