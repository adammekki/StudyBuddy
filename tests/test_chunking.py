import pymupdf

from studybuddy.chunking import extract_pages, make_chunks


def test_chunks_carry_source_page_and_skip_furniture():
    pages = [(1, "Short."), (2, "".join(f"Point {i} about the topic. " for i in range(120)))]
    chunks = make_chunks(pages, "deck.pdf")
    assert all(c["source"] == "deck.pdf" for c in chunks)
    assert all(len(c["text"]) > 40 for c in chunks)
    assert chunks[0]["page"] == 2
    assert chunks[0]["chunk_index"] == 0 and chunks[1]["chunk_index"] == 1
    assert chunks[0]["text"] != chunks[1]["text"]


def test_extract_pages_reads_page_numbers(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Lecture one content")
    path = tmp_path / "d.pdf"
    doc.save(path)
    doc.close()
    assert extract_pages(path)[0][0] == 1
    assert "Lecture one" in extract_pages(path)[0][1]
