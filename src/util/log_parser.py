from util.config_parser import TextParser

tp = TextParser()

class LogTable:
    def __init__(self):
        self.name = "Unknow"
        self.table = {}
        
    def parse(self, strlog):
        assert strlog[:2] == "**"
        assert strlog[-1] == "#"
        _strlog = strlog[2:-1]
        self.name, msg = _strlog.split("*")
        items = msg.split(" ")
        for item in items:
            key, value = item.split(":")
            self.table[key] = tp.trans_items(value)
    
    def __str__(self):
        header = "**%s*" % self.name
        msg = ' '.join([key+":"+str(self.table[key]) for key in self.table.keys()])
        log = header + msg + "#"
        return log


class LogParser:
    def __init__(self):
        self.logs = {}
        self.msg_squeue = []
    
    def resest(self):
        self.logs = {}
        self.msg_squeue = []
    
    def load(self, logpath):
        with open(logpath, "r") as f:
            for line in f:
                strlog = line.strip()
                if "nan" in strlog:
                    continue
                if "**" in strlog and "#" in strlog:
                    startid = strlog.find("**")
                    endid = strlog.find("#")
                    strlog = strlog[startid:endid+1]
                    tmptable = LogTable()
                    tmptable.parse(strlog)
                    if tmptable.name not in self.logs:
                        self.logs[tmptable.name] = []
                    self.logs[tmptable.name].append(tmptable)
                    self.msg_squeue.append(tmptable)
                    
            for key, item in self.logs.items():
                print(key, "len =",len(item))
                
    def append(self, name, table):
        if name not in self.logs:
            self.logs[name] = []
        new_log = LogTable()
        new_log.name = name
        new_log.table = table
        self.logs[name].append(new_log)
        
        s = " ".join([str(key)+":"+str(value) for key, value in table.items()])
        msg = "**%s*%s#\n" % (str(name), s)
        self.msg_squeue.append(msg)
    
    def save(self, logpath):
        with open(logpath, "w") as f:
            for i in range(len(self.msg_squeue)):
                f.write(self.msg_squeue[i])
        
        
if __name__=="__main__":
    pass


    
    
    