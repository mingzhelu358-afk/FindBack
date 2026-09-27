"""Single-camera temporal reasoning for simple object occlusion relationships."""

from __future__ import annotations

import time
from typing import Any


Box = tuple[int, int, int, int]


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def intersection_area(first: Box, second: Box) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def coverage_ratio(subject_box: Box, covering_box: Box) -> float:
    area = box_area(subject_box)
    if area <= 0:
        return 0.0
    return intersection_area(subject_box, covering_box) / area


class SpatialReasoner:
    """
    Infer that an object is probably behind a reference object.

    This first version deliberately uses temporal evidence instead of claiming
    physical depth from a single still image. It remembers a subject's last
    unobstructed box, then watches for a reference object covering that box
    while the subject shrinks or disappears.
    """

    def __init__(
        self,
        subjects: list[str],
        reference_name: str = "laptop",
        memory_seconds: float = 90.0,
    ) -> None:
        self.subjects = subjects
        self.reference_name = reference_name
        self.memory_seconds = memory_seconds
        self.tracks: dict[str, dict[str, Any]] = {}

    def update(
        self,
        detections: dict[str, dict[str, Any]],
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        if now is None:
            now = time.monotonic()

        reference = detections.get(self.reference_name)
        relations: list[dict[str, Any]] = []

        if reference is not None:
            reference_box = reference["box"]

            for subject_name in self.subjects:
                subject = detections.get(subject_name)
                track = self.tracks.get(subject_name)

                if not track:
                    continue

                age = now - track["last_seen"]
                if age > self.memory_seconds:
                    continue

                clear_box = track["clear_box"]
                clear_area = max(box_area(clear_box), 1.0)
                historical_coverage = coverage_ratio(
                    clear_box,
                    reference_box,
                )

                if subject is not None:
                    current_box = subject["box"]
                    current_area = box_area(current_box)
                    visible_ratio = min(
                        1.0,
                        current_area / clear_area,
                    )
                    current_overlap = coverage_ratio(
                        current_box,
                        reference_box,
                    )

                    # Strongest signal: a previously clear object becomes much
                    # smaller while the laptop covers its former position.
                    if (
                        historical_coverage >= 0.15
                        and visible_ratio <= 0.78
                    ):
                        confidence = min(
                            0.95,
                            0.52
                            + 0.28 * historical_coverage
                            + 0.25 * (1.0 - visible_ratio),
                        )
                        relations.append(
                            self._build_relation(
                                subject_name=subject_name,
                                location=subject["location"],
                                confidence=confidence,
                                visible_ratio=visible_ratio,
                                coverage=historical_coverage,
                                subject_box=current_box,
                                reference_box=reference_box,
                                fully_hidden=False,
                            )
                        )

                    # Secondary signal: the two current detection boxes overlap
                    # substantially and there is already temporal history.
                    elif current_overlap >= 0.24:
                        confidence = min(
                            0.80,
                            0.48 + 0.32 * current_overlap,
                        )
                        relations.append(
                            self._build_relation(
                                subject_name=subject_name,
                                location=subject["location"],
                                confidence=confidence,
                                visible_ratio=visible_ratio,
                                coverage=current_overlap,
                                subject_box=current_box,
                                reference_box=reference_box,
                                fully_hidden=False,
                            )
                        )

                # The subject disappeared, but the laptop now covers the last
                # clear position. Keep a cautious "probably behind" memory.
                elif historical_coverage >= 0.28:
                    confidence = min(
                        0.90,
                        0.56 + 0.32 * historical_coverage,
                    )
                    relations.append(
                        self._build_relation(
                            subject_name=subject_name,
                            location=track["location"],
                            confidence=confidence,
                            visible_ratio=0.0,
                            coverage=historical_coverage,
                            subject_box=clear_box,
                            reference_box=reference_box,
                            fully_hidden=True,
                        )
                    )

        relation_subjects = {
            relation["subject_name"]
            for relation in relations
        }

        # Update temporal memory only after evaluating the current frame, so a
        # small partially visible box does not overwrite the former clear box.
        for subject_name in self.subjects:
            subject = detections.get(subject_name)
            if subject is None:
                continue

            current_box = subject["box"]
            existing = self.tracks.get(subject_name)

            should_refresh_clear_box = (
                existing is None
                or subject_name not in relation_subjects
            )

            if should_refresh_clear_box:
                clear_box = current_box
            else:
                clear_box = existing["clear_box"]

            self.tracks[subject_name] = {
                "last_box": current_box,
                "clear_box": clear_box,
                "location": subject["location"],
                "confidence": subject["confidence"],
                "last_seen": now,
            }

        return relations

    def _build_relation(
        self,
        *,
        subject_name: str,
        location: str,
        confidence: float,
        visible_ratio: float,
        coverage: float,
        subject_box: Box,
        reference_box: Box,
        fully_hidden: bool,
    ) -> dict[str, Any]:
        visible_percent = round(visible_ratio * 100)
        coverage_percent = round(coverage * 100)

        if fully_hidden:
            evidence = (
                f"物品当前不可见；{self.reference_name}覆盖了其此前位置的"
                f"{coverage_percent}%"
            )
        else:
            evidence = (
                f"物品估计剩余可见面积{visible_percent}%；"
                f"{self.reference_name}覆盖其历史位置{coverage_percent}%"
            )

        return {
            "subject_name": subject_name,
            "relation": "behind",
            "reference_name": self.reference_name,
            "location": location,
            "confidence": round(confidence, 3),
            "visible_ratio": round(visible_ratio, 3),
            "evidence": evidence,
            "subject_box": subject_box,
            "reference_box": reference_box,
            "fully_hidden": fully_hidden,
        }
