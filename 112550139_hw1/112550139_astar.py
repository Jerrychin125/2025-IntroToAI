import csv
edgeFile = 'edges.csv'
heuristicFile = 'heuristic_values.csv'


def astar(start, end):
    # Begin your code (Part 4)
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
    # Build the heuristic
    heuristic = {}
    with open(heuristicFile, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            col_idx = header.index(str(end))
        except ValueError:
            raise ValueError("Heuristic for target node not found in header.")
        for row in reader:
            node = int(row[0])
            heuristic[node] = float(row[col_idx])
    # A* (priority queue)
    q = [(0, start, [start], 0)] # priority, current node, path, distance
    visited = []
    num_visited = 0
    while q:
        _, node, path, dist = q.pop(0)
        num_visited += 1

        if node == end:
            return path, dist, num_visited
        if node not in visited:
            visited.append(node)
            for next_node, weight in edges.get(node, {}).items():
                if next_node not in visited:
                    q.append((dist + weight + heuristic[next_node], next_node, path + [next_node], dist + weight))
                    q = sorted(q, key=lambda x: x[0])
    return [], 0, 0
    raise NotImplementedError("To be implemented")
    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
