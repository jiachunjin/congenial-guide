import json

Q = {
    "count": "How many {obj}s are there in the image?",
    "colors": "Is the {obj} in the image {color}? Directly answer yes or no, with no other words.",
    "position": "Is the {obj_2} {position} the {obj_1}? Directly answer yes or no, with no other words.",
}


with open("evaluation/generation/geneval/evaluation_metadata.jsonl") as fp:
    geneval_metadata = [json.loads(line) for line in fp]

# create a jsonl file to store correct answers
with open("evaluation/generation/geneval/correct_answers.jsonl", "w") as gt_file:
    for index, test_case in enumerate(geneval_metadata):
        index = f"{index:0>5}"
        if test_case["tag"] == "single_object":
            obj = test_case["include"][0]["class"]
            correct_count = test_case["include"][0]["count"]
            gt_file.write(json.dumps({
                "index": index,
                "question": Q["count"].format(obj=obj),
                "correct_answer": correct_count
            }) + "\n")

        if test_case["tag"] == "two_object":
            for count_case in test_case["include"]:
                obj = count_case["class"]
                correct_count = count_case["count"]
                question = Q["count"].format(obj=obj)
                gt_file.write(json.dumps({
                    "index": index,
                    "question": Q["count"].format(obj=obj),
                    "correct_answer": correct_count
                }) + "\n")

        if test_case["tag"] == "counting":
            for count_case in test_case["include"]:
                obj = count_case["class"]
                correct_count = count_case["count"]
                question = Q["count"].format(obj=obj)
                gt_file.write(json.dumps({
                    "index": index,
                    "question": Q["count"].format(obj=obj),
                    "correct_answer": correct_count
                }) + "\n")

        if test_case["tag"] == "colors":
            for color_case in test_case["include"]:
                obj = color_case["class"]
                color = color_case["color"]
                question = Q["colors"].format(obj=obj, color=color)
                print(index, question)
                gt_file.write(json.dumps({
                    "index": index,
                    "question": Q["colors"].format(obj=obj, color=color),
                    "correct_answer": "yes"
                }) + "\n")

        if test_case["tag"] == "position":
            obj_1 = test_case["include"][0]["class"]
            obj_2 = test_case["include"][1]["class"]
            position_relation = test_case["include"][1]["position"][0]
            question = Q["position"].format(obj_1=obj_1, obj_2=obj_2, position=position_relation)
            gt_file.write(json.dumps({
                "index": index,
                "question": question,
                "correct_answer": "yes"
            }) + "\n")

        if test_case["tag"] == "color_attr":
                for color_case in test_case["include"]:
                    obj = color_case["class"]
                    color = color_case["color"]
                    question = Q["colors"].format(obj=obj, color=color)

                    gt_file.write(json.dumps({
                        "index": index,
                        "question": question,
                        "correct_answer": "yes"
                    }) + "\n")