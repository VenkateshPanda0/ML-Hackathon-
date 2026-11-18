"""Folium map: GPS observations, nearby road network, reference route and predicted route."""
import folium
import numpy as np


def arcs_to_lines(trip, arc_ids):
    return [[(trip.nodes_ll[trip.arcs[a, 0], 1], trip.nodes_ll[trip.arcs[a, 0], 0]),
             (trip.nodes_ll[trip.arcs[a, 1], 1], trip.nodes_ll[trip.arcs[a, 1], 0])] for a in arc_ids]


def trip_map(trip, predicted=None, reference=None, title="", network_radius_m=150):
    lat, lon = trip.track_ll[:, 1], trip.track_ll[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=14, tiles="OpenStreetMap")
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    # road network near the trajectory only (full networks have up to 160k arcs)
    if trip.nodes_xy is not None:
        mid = (trip.nodes_xy[trip.arcs[:, 0]] + trip.nodes_xy[trip.arcs[:, 1]]) / 2
        from scipy.spatial import cKDTree
        d, _ = cKDTree(trip.track_xy[::5]).query(mid)
        near = np.where(d < network_radius_m)[0]
        net = folium.FeatureGroup(name="Road network (near track)", show=True)
        folium.PolyLine(arcs_to_lines(trip, near), color="#9e9e9e", weight=1, opacity=0.6).add_to(net)
        net.add_to(m)
    if reference is not None:
        g = folium.FeatureGroup(name="Reference route (ground truth)")
        folium.PolyLine(arcs_to_lines(trip, reference), color="#2ca02c", weight=7, opacity=0.6).add_to(g)
        g.add_to(m)
    if predicted is not None:
        g = folium.FeatureGroup(name="Predicted route (PathSnap)")
        folium.PolyLine(arcs_to_lines(trip, predicted), color="#d62728", weight=3, opacity=0.9).add_to(g)
        g.add_to(m)
    g = folium.FeatureGroup(name="GPS observations")
    step = max(1, len(lat) // 1500)
    for y, x in zip(lat[::step], lon[::step]):
        folium.CircleMarker([y, x], radius=2, color="#1f77b4", fill=True, fill_opacity=0.8, weight=0).add_to(g)
    g.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = f"""<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px 12px;
      border:1px solid #999;border-radius:6px;font-size:13px"><b>{title}</b><br>
      <span style="color:#1f77b4">&#9679;</span> GPS observations<br>
      <span style="color:#9e9e9e">&#9644;</span> Road network<br>
      <span style="color:#2ca02c">&#9644;</span> Reference route<br>
      <span style="color:#d62728">&#9644;</span> Predicted route</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m
