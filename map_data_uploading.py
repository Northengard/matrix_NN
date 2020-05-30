import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx


def save_json(data, filename):
    with open(filename, 'w') as out_file:
        out_file.write(data.to_json())


if __name__ == '__main__':
    # map uploading
    place_name = "Nizhny Novgorod, Russia"
    graph = ox.graph_from_place(place_name)

    nodes, edges = ox.graph_to_gdfs(graph)
    area = ox.gdf_from_place(place_name)
    buildings = ox.footprints_from_place(place_name)

    # saving
    nx.write_gpickle(graph, 'nizhny_novgorod_graph.pkl')
    save_json(edges, 'nn_roads.json')
    save_json(buildings, 'nn_buildings.json')
    save_json(nodes, 'nn_nodes.json')
    save_json(area, 'nn_area.json')

    fig, ax = plt.subplots(figsize=(20, 17))
    area.plot(ax=ax, facecolor='black')
    edges.plot(ax=ax, linewidth=1, edgecolor='#BC8F8F')
    buildings.plot(ax=ax, facecolor='khaki', alpha=0.7)
    plt.tight_layout()
    plt.title(place_name)
    plt.xlabel('lat')
    plt.ylabel('long')
    plt.show()
