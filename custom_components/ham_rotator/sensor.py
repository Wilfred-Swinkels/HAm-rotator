"""A self-computed wind-direction-average sensor.

Most weather station integrations already expose a 10-minute-average wind
direction sensor, so there's nothing to build for that case -- just point
the integration's "wind direction sensor (average)" option at it. This
optional sensor exists for any other averaging window the user wants
(entered directly in minutes), since that's rarely available anywhere
else.

Averaging a compass direction can't use a plain arithmetic mean -- e.g.
350 degrees and 10 degrees should average to 0, not 180. Each sample is
instead treated as a vector, the vectors are summed, and the mean
direction is read back off the resulting vector's angle (a standard
circular/vector mean).

Each direction sample's vector is weighted by the wind speed at that
moment (if a wind speed sensor is configured), rather than every sample
counting equally. Plain unweighted circular averaging gets unstable
during calm, variable wind -- the direction swings all over with nothing
holding it, so the resulting vector is short and its angle barely
meaningful, making the average jump around even though "the wind" barely
blew at all. Weighting by speed lets the (rarer, but real) stronger gusts
dominate the average instead of being swamped by noise from near-still
air.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from datetime import timedelta

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import DEGREE
from homeassistant.core import Event, HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.restore_state import ExtraStoredData, RestoreEntity

from .const import (
    CONF_WIND_AVG_MINUTES,
    CONF_WIND_DIRECTION_ENTITY,
    CONF_WIND_SPEED_ENTITY,
    DOMAIN,
    WIND_AVG_SUFFIX,
)

_LOGGER = logging.getLogger(__name__)

_RECOMPUTE_INTERVAL = timedelta(seconds=30)  # keeps the average decaying even when the source sensor is quiet


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the wind-direction-average sensor, if configured."""
    runtime = hass.data[DOMAIN][entry.entry_id]
    source_entity_id = runtime.client.get_option(CONF_WIND_DIRECTION_ENTITY)
    if not source_entity_id:
        return

    raw_minutes = runtime.client.get_option(CONF_WIND_AVG_MINUTES)
    try:
        minutes = float(raw_minutes)
    except (TypeError, ValueError):
        return
    if minutes <= 0:
        return

    speed_entity_id = runtime.client.get_option(CONF_WIND_SPEED_ENTITY)
    async_add_entities(
        [
            WindDirectionAverageSensor(
                entry,
                source_entity_id,
                speed_entity_id,
                minutes * 60,
                f"Wind direction ({minutes:g} min average)",
                WIND_AVG_SUFFIX,
            )
        ]
    )


class WindDirectionAverageExtraData(ExtraStoredData):
    """Sample buffer, persisted so a restart doesn't empty the average.

    Sample ages (seconds before the save) are stored rather than raw
    timestamps, since `time.monotonic()` -- used for everything else in
    this file -- isn't meaningful across a restart.
    """

    def __init__(self, samples: list[tuple[float, float, float]]) -> None:
        self.samples = samples

    def as_dict(self) -> dict:
        return {"samples": self.samples}

    @classmethod
    def from_dict(cls, restored: dict) -> "WindDirectionAverageExtraData":
        return cls(restored.get("samples", []))


class WindDirectionAverageSensor(SensorEntity, RestoreEntity):
    """Circular (vector) average of the realtime wind direction sensor."""

    _attr_has_entity_name = True
    _attr_native_unit_of_measurement = DEGREE
    _attr_icon = "mdi:compass-outline"

    def __init__(
        self,
        entry: ConfigEntry,
        source_entity_id: str,
        speed_entity_id: str | None,
        window_seconds: float,
        name: str,
        unique_suffix: str,
    ) -> None:
        self._source_entity_id = source_entity_id
        self._speed_entity_id = speed_entity_id
        self._window_seconds = window_seconds
        self._attr_name = name
        self._attr_unique_id = f"{entry.entry_id}_{unique_suffix}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)}, name=entry.title
        )
        # (timestamp, direction_degrees, weight)
        self._samples: deque[tuple[float, float, float]] = deque()
        self._attr_native_value = None

    @property
    def extra_restore_state_data(self) -> WindDirectionAverageExtraData:
        now = time.monotonic()
        return WindDirectionAverageExtraData(
            [[now - t, d, w] for t, d, w in self._samples]
        )

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()

        last_extra_data = await self.async_get_last_extra_data()
        if last_extra_data is not None:
            now = time.monotonic()
            for age, degrees, weight in WindDirectionAverageExtraData.from_dict(
                last_extra_data.as_dict()
            ).samples:
                self._samples.append((now - age, degrees, weight))
            self._recompute()

        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self._source_entity_id], self._async_source_changed
            )
        )
        self.async_on_remove(
            async_track_time_interval(
                self.hass, self._async_recompute, _RECOMPUTE_INTERVAL
            )
        )
        current = self.hass.states.get(self._source_entity_id)
        if current is not None:
            self._add_sample(current.state)
            self._recompute()

    async def _async_source_changed(self, event: Event) -> None:
        new_state = event.data.get("new_state")
        if new_state is None:
            return
        self._add_sample(new_state.state)
        self._recompute()
        self.async_write_ha_state()

    async def _async_recompute(self, _now) -> None:
        self._recompute()
        self.async_write_ha_state()

    def _current_weight(self) -> float:
        """Current wind speed, used to weight the next sample.

        Falls back to 1 (every sample counts equally) when there's no wind
        speed sensor configured, or its value can't be read right now.
        """
        if self._speed_entity_id is None:
            return 1.0
        speed_state = self.hass.states.get(self._speed_entity_id)
        if speed_state is None:
            return 1.0
        try:
            return max(float(speed_state.state), 0.0)
        except (TypeError, ValueError):
            return 1.0

    def _add_sample(self, raw_state) -> None:
        try:
            degrees = float(raw_state)
        except (TypeError, ValueError):
            return
        self._samples.append((time.monotonic(), degrees, self._current_weight()))

    def _recompute(self) -> None:
        cutoff = time.monotonic() - self._window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        if not self._samples:
            self._attr_native_value = None
            return

        sin_sum = sum(w * math.sin(math.radians(d)) for _, d, w in self._samples)
        cos_sum = sum(w * math.cos(math.radians(d)) for _, d, w in self._samples)
        if sin_sum == 0 and cos_sum == 0:
            # Either every sample happened to land exactly opposite another,
            # or every weight was 0 (dead calm the whole window) -- no
            # well-defined mean direction either way; keep the previous
            # value rather than guessing.
            return
        mean = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
        self._attr_native_value = round(mean, 1)
