import pymupdf

from studybuddy.captions import caption_pages, page_has_visual


def make_page(doc, with_image=False):
    page = doc.new_page()
    page.insert_text((72, 72), "Slide body text " * 8)
    if with_image:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 200), 0)
        pix.set_rect(pix.irect, (200, 60, 60))
        page.insert_image(pymupdf.Rect(100, 150, 400, 350), pixmap=pix)
    return page


def test_image_page_is_visual_and_text_page_is_not(tmp_path):
    plain_doc = pymupdf.open()
    make_page(plain_doc)
    assert page_has_visual(plain_doc[0]) is False

    image_doc = pymupdf.open()
    make_page(image_doc, with_image=True)
    assert page_has_visual(image_doc[0]) is True


def test_captions_are_cached_per_file_hash(tmp_path):
    doc = pymupdf.open()
    make_page(doc, with_image=True)
    path = tmp_path / "deck.pdf"
    doc.save(path)
    doc.close()

    calls = {"n": 0}

    def captioner(png):
        calls["n"] += 1
        return "a red block diagram"

    first = caption_pages(path, "abc123", tmp_path / "cache", captioner)
    assert first[1] == "a red block diagram"
    caption_pages(path, "abc123", tmp_path / "cache", captioner)
    assert calls["n"] == 1


def test_caption_is_capped(tmp_path):
    doc = pymupdf.open()
    make_page(doc, with_image=True)
    path = tmp_path / "deck.pdf"
    doc.save(path)
    doc.close()

    out = caption_pages(path, "sha", tmp_path / "cache", lambda png: "x" * 1000)
    assert len(out[1]) <= 400
