classIDs = []
LABELS = []
stain_size = []
x = y = z = 0
total_stain_size_x = total_stain_size_y = total_stain_size_z = 0
set_stain = 10000

Manual = [{
    "key": 0,
    "detergent": "중성세제 반컵 사용",
    "temp": 30,
    "washCycle": "헹굼 2회 + 탈수 중"
}]


if len(idxs) > 0:
    for i in idxs.flatten():
        if LABELS[classIDs[i]] == 'Swimwear':
            y += 1
            total_stain_size_y += stain_size[i]
        elif LABELS[classIDs[i]] == 'Towel':
            z += 1
            total_stain_size_z += stain_size[i]
        else:
            x += 1
            total_stain_size_x += stain_size[i]

if y != 0:
    Manual.append({
        "key": 1,
        "detergent": "중성세제 반컵 사용, 염소계표백제 사용 금지",
        "temp": 40,
        "washCycle": "헹굼 2회 + 탈수 약 + 건조 금지"
    })
    if total_stain_size_y > set_stain:
        if Manual[1]["key"] == 1:
            Manual[1]["washCycle"] = "헹굼 3회 + 탈수 약 + 건조 금지 + 오염도 높음"
if z != 0:
    Manual.append({
        "key": 2,
        "detergent": "중성세제 반컵 사용, 섬유유연제 & 염소계표백제 사용 금지",
        "temp": 40,
        "washCycle": "헹굼 2회 + 탈수 약 + 건조 금지"
    })
    if total_stain_size_z > set_stain:
        if Manual[1]["key"] == 2:
            Manual[1]["washCycle"] = "헹굼 3회 + 탈수 약 + 오염도 높음"
        elif Manual[2]["key"] == 2:
            Manual[2]["washCycle"] = "헹굼 3회 + 탈수 약 + 오염도 높음"
if x > 7:
    Manual[0] = {
        "key": 3,
        "detergent": "중성세제 한컵 사용",
        "temp": 40,
        "washCycle": "헹굼 3회 + 탈수 중"
    }
    if total_stain_size_x > set_stain:
        Manual[0]["washCycle"] = "헹굼 4회 + 탈수 중 + 오염도 높음"
elif x <= 7:
    if total_stain_size_x > set_stain:
        Manual[0]["washCycle"] = "헹굼 3회 + 탈수 중 + 오염도 높음"

print(Manual)
