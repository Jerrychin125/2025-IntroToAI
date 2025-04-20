import csv
edgeFile = 'edges.csv'


def dfs(start, end):
    # Begin your code (Part 2)
    # Build the graph
    edges = {}
    with open(edgeFile, 'r', encoding='utf-8') as f:
        row = csv.reader(f)
        next(row)
        for s, t, w in row:
            s, t, w = int(s), int(t), float(w)
            if s not in edges:
                edges[s] = {}
            edges[s][t] = w
    # DFS (stack)
    s = [(start, [start], 0)] # current node, path, distance
    visited = []
    num_visited = 0
    while s:
        node, path, dist = s.pop()
        num_visited += 1

        if node == end:
            return path, dist, num_visited
        if node not in visited:
            visited.append(node)
            for next_node, weight in edges.get(node, {}).items():
                if next_node not in visited:
                    s.append((next_node, path + [next_node], dist + weight))
    return [], 0, 0

    raise NotImplementedError("To be implemented")
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
