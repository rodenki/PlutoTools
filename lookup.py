import socket
import numpy as np
from socket import error as socket_error

class Scanner:

    def __init__(self):
        self.addresses = []
        self.hostnames = []

    def scanHostNames(self, ranges):
        self.addresses = self.supIPs(ranges)
        self.hostnames = np.array(self.getHostNames(self.addresses), dtype=object)

    def getHostNames(self, addresses):
        hostnames = []
        for address in addresses:
            try:
                response = socket.gethostbyaddr(address)
                hostnames.append([response[0], response[2][0]])
            except socket_error:
                print(address + " not found")
        return hostnames

    def supIPs(self, ranges):
        addresses = []
        for a in ranges[0]:
            for b in ranges[1]:
                for c in ranges[2]:
                    for d in ranges[3]:
                        addresses.append(str(a) + "." + str(b) + "." + str(c) + "." + str(d))
        return addresses


scanner = Scanner()
ranges = [range(149, 150), range(217, 218), range(0, 256), range(0, 256)]
scanner.scanHostNames(ranges)
f = open("output.dat", 'w')
for item in scanner.hostnames:
    f.write(item[1] + "\t" + item[0] + "\n")

f.close()
