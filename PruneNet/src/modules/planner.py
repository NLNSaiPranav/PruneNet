"""
planner.py

Global path planning module.

Pipeline

Localized Trees
        │
        ▼
Tree Obstacles
        │
        ▼
Voronoi Diagram
        │
        ▼
Navigation Graph
        │
        ▼
Elkai TSP
        │
        ▼
Pruning Path
"""

import os
import cv2
import json
import elkai
import numpy as np
import networkx as nx
import pandas as pd

from shapely.geometry import (
    Point,
    Polygon,
    MultiPolygon,
    LineString
)

from shapely.ops import (
    unary_union,
    voronoi_diagram
)

from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt

from matplotlib.patches import (
    Circle,
    Rectangle
)
from src.core.database import TreeDatabase
class OrchardPlanner:

    def __init__(

        self,

        obstacle_scale=0.8,

        graph_resolution=1.0

    ):

        self.obstacle_scale = obstacle_scale

        self.graph_resolution = graph_resolution
    # =====================================================
    # Orchard Boundary
    # =====================================================

    def create_boundary(

        self,

        orthomosaic

    ):

        white_mask = cv2.inRange(

            orthomosaic,

            np.array([247,247,247]),

            np.array([255,255,255])

        )

        non_white = cv2.bitwise_not(
            white_mask
        )

        kernel = np.ones(
            (5,5),
            np.uint8
        )

        non_white = cv2.dilate(

            non_white,

            kernel,

            iterations=1

        )

        contours, _ = cv2.findContours(

            non_white,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE

        )

        largest = max(
            contours,
            key=cv2.contourArea
        )

        epsilon = (
            0.0005 *
            cv2.arcLength(
                largest,
                True
            )
        )

        approx = cv2.approxPolyDP(

            largest,

            epsilon,

            True

        ).squeeze()

        if not np.allclose(

            approx[0],

            approx[-1],

            atol=10

        ):

            approx = np.vstack(

                [

                    approx,

                    approx[0]

                ]

            )

        return Polygon(approx)
    # =====================================================
    # Tree Coordinates
    # =====================================================

    def extract_tree_geometry(

        self,

        database: TreeDatabase

    ):

        centers = []

        radii = []

        for tree in database:

            if tree.is_duplicate:
                continue

            if tree.orthomosaic_center is None:
                continue

            centers.append(

                tree.orthomosaic_center

            )

            radii.append(

                tree.orthomosaic_radius

            )

        return (

            np.asarray(centers),

            np.asarray(radii)

        )
    # =====================================================
    # Pruning Targets
    # =====================================================

    def pruning_targets(

        self,

        database

    ):

        targets = []

        for tree in database:

            if not tree.needs_pruning:
                continue

            if tree.is_duplicate:
                continue

            if tree.orthomosaic_center is None:
                continue

            targets.append(tree)

        return targets
    # =====================================================
    # Union Find
    # =====================================================

    def find(
        self,
        parent,
        x
    ):

        while parent[x] != x:

            parent[x] = parent[parent[x]]

            x = parent[x]

        return x


    def union(
        self,
        parent,
        x,
        y
    ):

        root_x = self.find(
            parent,
            x
        )

        root_y = self.find(
            parent,
            y
        )

        if root_x != root_y:

            parent[root_y] = root_x
    # =====================================================
    # Cluster Overlapping Trees
    # =====================================================

    def cluster_trees(
        self,
        centers,
        radii
    ):

        parent = list(
            range(
                len(centers)
            )
        )

        for i in range(len(centers)):

            for j in range(i + 1, len(centers)):

                distance = np.linalg.norm(

                    centers[i] -
                    centers[j]

                )

                if distance < (

                    radii[i] +
                    radii[j]

                ):

                    self.union(

                        parent,

                        i,

                        j

                    )

        clusters = {}

        for i in range(len(centers)):

            root = self.find(
                parent,
                i
            )

            clusters.setdefault(
                root,
                []
            ).append(i)

        return list(
            clusters.values()
        )
    # =====================================================
    # Build Obstacles
    # =====================================================

    def create_obstacles(
        self,
        centers,
        radii,
        clusters
    ):

        generator_points = []

        obstacle_shapes = []

        metadata = []

        for cluster in clusters:

            cluster_centers = centers[cluster]

            cluster_radii = radii[cluster]

            if len(cluster) == 1:

                x, y = cluster_centers[0]

                r = cluster_radii[0]

                generator_points.append(
                    Point(x, y)
                )

                obstacle_shapes.append(

                    Point(x, y).buffer(

                        r *
                        self.obstacle_scale

                    )

                )

                metadata.append({

                    "type": "single",

                    "x": float(x),

                    "y": float(y),

                    "radius": float(r)

                })

            else:

                xmin = np.min(
                    cluster_centers[:,0] -
                    cluster_radii
                )

                xmax = np.max(
                    cluster_centers[:,0] +
                    cluster_radii
                )

                ymin = np.min(
                    cluster_centers[:,1] -
                    cluster_radii
                )

                ymax = np.max(
                    cluster_centers[:,1] +
                    cluster_radii
                )

                cx = (

                    xmin +
                    xmax

                ) / 2

                cy = (

                    ymin +
                    ymax

                ) / 2

                generator_points.append(

                    Point(cx, cy)

                )

                obstacle_shapes.append(

                    Polygon([

                        (xmin, ymin),

                        (xmax, ymin),

                        (xmax, ymax),

                        (xmin, ymax)

                    ])

                )

                metadata.append({

                    "type": "cluster",

                    "x": float(cx),

                    "y": float(cy),

                    "members": cluster

                })

        return (

            generator_points,

            obstacle_shapes,

            metadata

        )
    # =====================================================
    # Create Voronoi Diagram
    # =====================================================

    def create_voronoi(
        self,
        boundary,
        generator_points
    ):

        generators = unary_union(
            generator_points
        )

        vor = voronoi_diagram(

            generators,

            envelope=boundary,

            tolerance=0.1

        )

        return vor

    # =====================================================
    # Navigation Graph
    # =====================================================

    def build_graph(
        self,
        voronoi,
        boundary,
        obstacles
    ):

        graph = nx.Graph()

        for poly in voronoi.geoms:

            clipped = poly.intersection(
                boundary
            )

            polygons = []

            if isinstance(
                clipped,
                Polygon
            ):

                polygons.append(clipped)

            elif isinstance(
                clipped,
                MultiPolygon
            ):

                polygons.extend(
                    clipped.geoms
                )

            else:

                continue

            for polygon in polygons:

                coords = np.array(
                    polygon.exterior.coords
                )

                for i in range(
                    len(coords) - 1
                ):

                    p1 = tuple(coords[i])

                    p2 = tuple(coords[i + 1])

                    edge = LineString(
                        [p1, p2]
                    )

                    collision = False

                    for obstacle in obstacles:

                        if edge.intersects(
                            obstacle
                        ):

                            collision = True

                            break

                    if collision:

                        continue
                    distance = np.linalg.norm(

                        np.asarray(p1)

                        -

                        np.asarray(p2)

                    )

                    graph.add_edge(

                        p1,

                        p2,

                        weight=distance

                    )
        return graph
    # =====================================================
    # Closest Graph Node
    # =====================================================

    def closest_node(
        self,
        point,
        graph
    ):

        nodes = list(
            graph.nodes
        )

        point = np.asarray(
            point
        )

        distances = [

            np.linalg.norm(

                np.asarray(node)

                -

                point

            )

            for node in nodes

        ]

        return nodes[
            np.argmin(
                distances
            )
        ]
    # =====================================================
    # Compute Navigation Paths
    # =====================================================

    def compute_paths(
        self,
        graph,
        targets
    ):

        shortest_paths = {}

        centers = [

            tree.orthomosaic_center

            for tree in targets

        ]

        for i in range(len(centers)):

            for j in range(i + 1, len(centers)):

                source = self.closest_node(

                    centers[i],

                    graph

                )

                target = self.closest_node(

                    centers[j],

                    graph

                )

                try:

                    path = nx.shortest_path(

                        graph,

                        source,

                        target,

                        weight="weight"

                    )

                    length = nx.shortest_path_length(

                        graph,

                        source,

                        target,

                        weight="weight"

                    )

                    shortest_paths[
                        (i, j)
                    ] = (

                        path,

                        length

                    )

                except:

                    continue

        return shortest_paths
    # =====================================================
    # Distance Matrix
    # =====================================================

    def create_distance_matrix(
        self,
        shortest_paths,
        n
    ):

        D = np.zeros(
            (n, n)
        )

        for (i, j), (_, dist) in shortest_paths.items():

            D[i, j] = dist

            D[j, i] = dist

        return D
    # =====================================================
    # Elkai TSP
    # =====================================================

    def solve_tsp(
        self,
        distance_matrix
    ):

        matrix = distance_matrix.astype(int).tolist()

        order = elkai.solve_int_matrix(
            matrix
        )

        return order
    
    # =====================================================
    # Reconstruct Route
    # =====================================================

    def reconstruct_route(
        self,
        tsp_order,
        shortest_paths
    ):

        route = []

        for i in range(
            len(tsp_order)-1
        ):

            a = tsp_order[i]
            b = tsp_order[i+1]

            key = (
                min(a,b),
                max(a,b)
            )

            if key not in shortest_paths:
                continue

            path, _ = shortest_paths[key]

            if a > b:
                path = path[::-1]

            if len(route) == 0:
                route.extend(path)
            else:
                route.extend(path[1:])

        return route
    
    # =====================================================
    # Update Visit Order
    # =====================================================

    def update_database(
        self,
        targets,
        tsp_order
    ):

        for visit, index in enumerate(
            tsp_order
        ):

            targets[index].visit_order = visit

        return targets
    
    # =====================================================
    # Draw Route
    # =====================================================

    def draw_route(
        self,
        image,
        database,
        boundary,
        vor,
        graph,
        route,
        output_folder
    ):

        output_dir = os.path.join(
            output_folder,
            "planner"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        canvas = np.full_like(
            image,
            255,
            dtype=np.uint8
        )

        # =====================================
        # Draw Voronoi Diagram
        # =====================================

        for cell in vor.geoms:

            if cell.is_empty:
                continue

            clipped = cell.intersection(boundary)

            polygons = []

            if isinstance(clipped, Polygon):

                polygons.append(clipped)

            elif isinstance(clipped, MultiPolygon):

                polygons.extend(clipped.geoms)

            for polygon in polygons:

                pts = np.array(

                    polygon.exterior.coords,

                    dtype=np.int32

                )

                cv2.polylines(

                    canvas,

                    [pts],

                    True,

                    (180,180,180),

                    1,

                    cv2.LINE_AA

                )

        # =====================================
        # Orchard Boundary
        # =====================================

        boundary_pts = np.array(
            boundary.exterior.coords,
            dtype=np.int32
        )

        cv2.polylines(
            canvas,
            [boundary_pts],
            True,
            (0, 0, 255),
            2
        )

        overlay = canvas.copy()

        for tree in database:

            if tree.orthomosaic_center is None:
                continue

            center = (
                int(tree.orthomosaic_center[0]),
                int(tree.orthomosaic_center[1])
            )

            radius = max(
                5,
                int(tree.orthomosaic_radius)
            )

            if tree.needs_pruning:
                color = (0,0,255)
            else:
                color = (120,220,120)

            cv2.circle(
                overlay,
                center,
                radius,
                color,
                -1
            )

        cv2.addWeighted(
            overlay,
            0.35,
            canvas,
            0.65,
            0,
            canvas
        )

        # =====================================
        # Elkai Route
        # =====================================

        for i in range(len(route) - 1):

            p1 = tuple(
                np.int32(route[i])
            )

            p2 = tuple(
                np.int32(route[i + 1])
            )

            cv2.line(
                canvas,
                p1,
                p2,
                (0, 165, 255),
                4,
                cv2.LINE_AA
            )
        # =====================================
        # Highlight pruning trees
        # =====================================

        for tree in database:

            if not tree.needs_pruning:
                continue

            if tree.orthomosaic_center is None:
                continue

            center = (
                int(tree.orthomosaic_center[0]),
                int(tree.orthomosaic_center[1])
            )

            cv2.circle(
                canvas,
                center,
                6,
                (0,0,255),
                -1
            )
        # ----------------------------------------
        # Save full resolution image
        # ----------------------------------------

        full_path = os.path.join(
            output_dir,
            "elkai_path_full.png"
        )

        cv2.imwrite(
            full_path,
            canvas
        )

        # ----------------------------------------
        # Save preview image
        # ----------------------------------------

        preview = cv2.resize(
            canvas,
            None,
            fx=0.15,
            fy=0.15,
            interpolation=cv2.INTER_AREA
        )

        preview_path = os.path.join(
            output_dir,
            "elkai_path.png"
        )

        cv2.imwrite(
            preview_path,
            preview
        )

        return preview_path
    
    # =====================================================
    # Export Planner
    # =====================================================

    def export_results(
        self,
        targets,
        tsp_order,
        output_folder
    ):

        rows = []

        for visit, idx in enumerate(
            tsp_order
        ):

            tree = targets[idx]

            rows.append({

                "visit_order": visit,

                "tree_id": tree.tree_id,

                "opening_percentage": tree.opening_percentage,

                "center_x": tree.orthomosaic_center[0],

                "center_y": tree.orthomosaic_center[1]

            })

        df = pd.DataFrame(rows)

        df.to_excel(

            os.path.join(

                output_folder,

                "planner",

                "elkai_route.xlsx"

            ),

            index=False

        )
    # =====================================================
    # Run
    # =====================================================

    def run(
        self,
        database,
        orthomosaic,
        output_folder
    ):

        boundary = self.create_boundary(
            orthomosaic
        )

        centers, radii = self.extract_tree_geometry(
            database
        )

        targets = self.pruning_targets(
            database
        )

        clusters = self.cluster_trees(
            centers,
            radii
        )

        generators, obstacles, metadata = self.create_obstacles(
            centers,
            radii,
            clusters
        )

        # -------- Voronoi from ALL trees --------

        # =====================================================
        # Planning Voronoi (clustered obstacles)
        # =====================================================

        planning_vor = self.create_voronoi(
            boundary,
            generators
        )

        graph = self.build_graph(
            planning_vor,
            boundary,
            obstacles
        )

        # =====================================================
        # Visualization Voronoi (all detected trees)
        # =====================================================

        vor_points = [

            Point(x, y)

            for x, y in centers

        ]

        visualization_vor = self.create_voronoi(
            boundary,
            vor_points
        )

        shortest_paths = self.compute_paths(
            graph,
            targets
        )

        D = self.create_distance_matrix(
            shortest_paths,
            len(targets)
        )

        tsp = self.solve_tsp(
            D
        )

        route = self.reconstruct_route(
            tsp,
            shortest_paths
        )

        self.update_database(
            targets,
            tsp
        )

        self.draw_route(
            orthomosaic,
            database,
            boundary,
            visualization_vor,
            graph,
            route,
            output_folder
        )

        self.export_results(
            targets,
            tsp,
            output_folder
        )

        return database