
class node:
    def __init__(self,char):
        self.char = char
        self.nodes = {}
        self.end = False

class Trie:
    def __init__(self):
        self.head = node('0')
        self.size = 1

    def insert(self,data):
        current_node = self.head
        for i in data:
            if i not in current_node.nodes:
                self.size += 1
                current_node.nodes[i] = node(i)
            current_node = current_node.nodes[i]
        current_node.end = True


t = Trie()
l = ['et','eq','bd','be','bdp']

for i in l:
    t.insert(i)
print(t.size)
