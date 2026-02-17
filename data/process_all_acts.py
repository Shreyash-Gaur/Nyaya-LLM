import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACT_FILES = {
    "ipc.json": "Indian Penal Code, 1860",
    "crpc.json": "Code of Criminal Procedure, 1973",
    "cpc.json": "Code of Civil Procedure, 1908",
    "iea.json": "Indian Evidence Act, 1872",
    "ida.json": "Indian Divorce Act, 1869",
    "hma.json": "Hindu Marriage Act, 1955",
    "nia.json": "Negotiable Instruments Act, 1881",
    "MVA.json": "Motor Vehicles Act, 1988"
}


def normalize_section(item, act_name):
    section = item.get("section") or item.get("Section") or ""
    chapter = item.get("chapter", "")
    chapter_title = item.get("chapter_title", "")

    title = item.get("title") or item.get("section_title") or ""
    description = item.get("description") or item.get("section_desc") or ""

    if isinstance(description, str):
        description = description.strip()

    return {
        "act_name": act_name,
        "chapter": chapter,
        "chapter_title": chapter_title,
        "section": section,
        "title": title,
        "description": description
    }


def create_samples(data):
    samples = []

    act = data["act_name"]
    section = data["section"]
    chapter = data["chapter"]
    chapter_title = data["chapter_title"]
    title = data["title"]
    description = data["description"]

    base_text = (
        f"{act}\n"
        f"Chapter {chapter}: {chapter_title}\n"
        f"Section {section}: {title}\n\n"
        f"{description}"
    )

    # 1️⃣ Direct Explanation
    samples.append({
        "instruction": f"Explain Section {section} of the {act}.",
        "input": "",
        "output": base_text
    })

    # 2️⃣ Simple Explanation
    samples.append({
        "instruction": f"Explain Section {section} ({title}) in simple terms.",
        "input": "",
        "output": f"In simple words, Section {section} of the {act} deals with {title}. {description}"
    })

    # 3️⃣ Legal Q&A
    samples.append({
        "instruction": f"What does Section {section} of {act} state?",
        "input": "",
        "output": description
    })

    # 4️⃣ Summarization
    samples.append({
        "instruction": f"Summarize Section {section} of {act}.",
        "input": "",
        "output": description[:400]
    })

    # 5️⃣ Act + Section Identification
    samples.append({
        "instruction": f"Under which Act does Section {section} titled '{title}' fall?",
        "input": "",
        "output": f"Section {section} titled '{title}' falls under the {act}."
    })

    return samples


def main():
    all_samples = []

    for file_name, act_name in ACT_FILES.items():
        file_path = RAW_DIR / file_name

        if not file_path.exists():
            print(f"Skipping missing file: {file_name}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            normalized = normalize_section(item, act_name)

            # 🔎 Cleaning
            if not normalized["description"]:
                continue

            if "omitted" in normalized["description"].lower():
                continue

            if len(normalized["description"]) < 50:
                continue

            all_samples.extend(create_samples(normalized))

    random.shuffle(all_samples)

    train, temp = train_test_split(all_samples, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    for split_name, split_data in zip(
        ["train", "val", "test"], [train, val, test]
    ):
        with open(OUTPUT_DIR / f"all_acts_{split_name}.jsonl", "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")

    print("All Acts dataset processing complete.")
    print(f"Total samples: {len(all_samples)}")
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")


if __name__ == "__main__":
    main()
