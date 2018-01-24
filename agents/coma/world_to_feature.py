#!/usr/bin/python
# -*- coding: UTF-8 -*-

from pysc2.lib import point
from pysc2.lib import transform

class World_To_Feature(object):
    def __init__(self):
        self._feature_layer_screen_size = point.Point(84.0,84.0)
        self._camera_width_world_units = 24.0
        self._map_size = point.Point(64, 64)
        self._camera_center = point.Point(33.0,25.0)

        self._world_to_screen = transform.Linear(point.Point(1, -1),
                                             point.Point(0, self._map_size.y))
        self._screen_to_fl_screen = transform.Linear(
            self._feature_layer_screen_size / self._camera_width_world_units)
        self._world_to_fl_screen = transform.Chain(
            self._world_to_screen,
            self._screen_to_fl_screen,
            transform.Floor())
        self._update_camera(self._camera_center)

    def _update_camera(self, camera_center):
        """Update the camera transform based on the new camera center."""
        camera_radius = (self._feature_layer_screen_size /
                         self._feature_layer_screen_size.x *
                         self._camera_width_world_units / 2)
        center = camera_center.bound(camera_radius, self._map_size - camera_radius)
        self._camera = point.Rect(
            (center - camera_radius).bound(self._map_size),
            (center + camera_radius).bound(self._map_size))
        self._world_to_screen.offset = (-self._camera.bl *
                                        self._world_to_screen.scale)

    def word_to_screen(self, x,y):
        return self._world_to_fl_screen.fwd_pt(point.Point(x,y))

world_to_feature = World_To_Feature()
