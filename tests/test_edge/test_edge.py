from server.edge.edge_initialization import Edge

def test_edge():
    e = Edge()
    height = 96
    width = 96
    e.initialization(input_height=height, input_width=width)