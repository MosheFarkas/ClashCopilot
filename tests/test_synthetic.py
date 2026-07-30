from clash_copilot.synthetic import (
    FRAME_SHAPE,
    HOG_CYCLE,
    PLAY_ZONE_PX,
    TILE,
    layout_dict,
    make_card_art,
    render_frames,
)


def test_card_art_tiles_are_distinct_and_sized():
    art = make_card_art(["A", "B"])
    assert art["A"].shape == (TILE, TILE, 3)
    assert not (art["A"] == art["B"]).all()


def test_render_frames_places_card_art_at_play_time():
    art = make_card_art(HOG_CYCLE.deck)
    fps = 5
    frames = render_frames(HOG_CYCLE, art, fps=fps)
    assert len(frames) == int(HOG_CYCLE.duration * fps)

    play_t, card = HOG_CYCLE.plays[0]
    x, y = PLAY_ZONE_PX
    crop = frames[int(play_t * fps)].image[y : y + TILE, x : x + TILE]
    assert (crop == art[card]).all()
    assert (frames[0].image == 40).all()  # background only before any play


def test_layout_dict_matches_render_geometry():
    d = layout_dict()
    height, width = FRAME_SHAPE[:2]
    x, y = PLAY_ZONE_PX
    assert d["play_zone"] == {
        "x": x / width, "y": y / height, "w": TILE / width, "h": TILE / height,
    }
    assert d["card_size"] == {"w": TILE / width, "h": TILE / height}
    assert 0 < d["threshold"] <= 1
