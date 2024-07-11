import threading, queue
import time

class CommandsAndCallbackThread(threading.Thread):
    def __init__(self, run=True, *args, **kwargs):
        self.queue = queue.Queue()
        self.stop=False
        super(CommandsAndCallbackThread, self).__init__(target=self.target_with_callback, args=(self.queue,), *args, **kwargs)
        if(run):
            self.start()

    def enqueue(self, func, cb=None, repeat = -1, **kwargs):
        self.queue.put_nowait({"func":func, "cb": cb, "args": kwargs, "repeat": repeat})
    
    def ask_stop(self):
        # with self.queue.mutex:
        #     try:
        #         while True:
        #             self.queue.get()
        #     except queue.Queue.Empty:
        #         pass
        #self.queue.put_nowait("<STOP>")
        self.stop = True

    def wait_stop(self):
        if(self.is_alive()):
            self.ask_stop()
        if(self.is_alive()):            
            self.join(1)

    def _execute_item(self, item):
        r = None
        if("func" in item and item["func"] is not None):
            args = item.get("args",{})
            r = item["func"](**args)
        if item["cb"] is not None:
            item["cb"](r)

    def target_with_callback(self, q):
        while(not self.stop):
            try:
                item = q.get(block=False)
                #print(f"thread received {item}")
                if(isinstance(item, str) and item == "<STOP>"):
                    break
                else:
                    self._execute_item(item)
                    if(item["repeat"]>0):
                        time.sleep(item["repeat"]*0.001)
                        self.enqueue(item["func"],item["cb"],item["repeat"], **item["args"])
                q.task_done()
            except Exception as e:
                pass
