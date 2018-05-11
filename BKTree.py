class Node(object):
    def __init__(self, content):
        self.content = content
        self.children = {}

    def __str__(self):
        return self.content


class Tree(object):
    def __init__(self, distance_func, words=None):
        self.root = None
        self.distance = distance_func
        if words:
            for word in words:
                self.add(word)

    def add(self, word):
        if self.root is None:
            self.root = Node(word)
        else:
            node = Node(word)
            curr = self.root
            distance = self.distance(word, curr.content)

            while distance in curr.children:
                curr = curr.children[distance]
                distance = self.distance(word, curr.content)

            curr.children[distance] = node
            node.parent = curr

    def search(self, word, max_distance):
        candidates = [self.root]
        found = []

        while len(candidates) > 0:
            node = candidates.pop(0)
            distance = self.distance(node.content, word)

            if distance <= max_distance:
                found.append(node)

            candidates.extend(child_node for child_dist, child_node in node.children.items()
                              if distance - max_distance <= child_dist <= distance + max_distance)

        return found