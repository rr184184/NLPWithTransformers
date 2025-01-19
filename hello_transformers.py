import time
import re

from transformers import pipeline
import pandas as pd

text = """Dear Amazon, last week I ordered an Optimus Prime action figure
from your online store in Germany. Unfortunately, when I opened the package,
I discovered to my horror that I had been sent an action figure of Megatron
instead! As a lifelong enemy of the Decepticons, I hope you can understand my
dilemma. To resolve the issue, I demand an exchange of Megatron for the
Optimus Prime figure I ordered. Enclosed are copies of my records concerning
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

def parse_comma_separated_text_file(file_path):
    with open(file_path, 'r') as f:
        contents = f.read()
    sentences = re.findall(r'"([^"]*)"', contents)
    [print(f"sentence:\n{sentence}\n") for sentence in sentences]
    return sentences

restaurant_reviews_file = "./resources/text/restaurant_reviews.txt"
restaurant_reviews_sentences = parse_comma_separated_text_file(restaurant_reviews_file)

def test_classifier():
    start = time.perf_counter()
    model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    classifier = pipeline(task="text-classification",
                          model=model,
                          device="cpu")
    outputs = classifier(restaurant_reviews_sentences)
    end = time.perf_counter()

    df = pd.DataFrame(outputs)
    for i in range(len(df)):
        print(f"sentiment: {df.iloc[i].get("label")}, score: {df.iloc[i].get("score")}")
    print(f"Classifier executed in {end - start} seconds using device: {classifier.device}.\n")

def test_ner():
    start = time.perf_counter()
    model = "dbmdz/bert-large-cased-finetuned-conll03-english"
    revision = "4c53496"
    ner_tagger = pipeline(task="ner",
                          aggregation_strategy="simple",
                          model=model,
                          revision=revision,
                          device="cuda:0")
    outputs = ner_tagger(text)
    end = time.perf_counter()

    df = pd.DataFrame(outputs)
    print(df)
    print(f"NER tagger executed in {end - start} seconds using device: {ner_tagger.device}.\n")

def test_question_answering():
    start = time.perf_counter()
    model = "distilbert/distilbert-base-cased-distilled-squad"
    revision = "564e9b5"
    qa = pipeline(task="question-answering",
                  model=model,
                  revision=revision,
                  device="cuda:0")
    questions = ["What does the customer want?",
                 "What is the customer complaining about?",
                 "What did the customer find out?",
                 "What did the user originally order?",
                 "What did the user actually get?",]
    end = time.perf_counter()

    for question in questions:
        output = qa(question=question, context=text)
        df = pd.DataFrame([output])
        print(f"question: {question}\nanswer: {df}")
    print(f"Question-Answering executed in {end - start} seconds using device: {qa.device}.\n")

def test_summarization():
    start = time.perf_counter()
    model = "sshleifer/distilbart-cnn-12-6"
    revision = "a4f8f3e"
    summarizer = pipeline(task="summarization",
                          model=model,
                          revision=revision,
                          device="cuda:0")
    outputs = summarizer(text,
                         min_length=35,
                         max_length=35,
                         clean_up_tokenization_spaces=True)
    end = time.perf_counter()

    print(f"summary:\n{outputs[0]['summary_text']}")
    print(f"Summarizer executed in {end - start} seconds using device: {summarizer.device}.\n")

def test_translation():
    start = time.perf_counter()
    model = "Helsinki-NLP/opus-mt-en-fr"
    revision = "dd7f654"
    translator = pipeline(task="translation_en_to_fr",
                          model=model,
                          # revision=revision,
                          device="cpu")
    outputs = translator(text,
                         min_length=100,
                         clean_up_tokenization_spaces=True)
    end = time.perf_counter()

    print(f"translation:\n{outputs[0]['translation_text']}")
    print(f"Translator executed in {end - start} seconds using device: {translator.device}.\n")

def test_text_generation():
    start = time.perf_counter()
    model = "openai-community/gpt2" # unhelpful
    # model = "bigscience/bloom" too large
    # model = "databricks/dolly-v2-12b"
    # revision = "607a30d"
    generator = pipeline(task="text-generation",
                         model=model,
                         # revision=revision,
                         trust_remote_code=True,
                         device="cpu",)
    response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
    prompt = text + "\n\nCustomer service response:\n" + response
    outputs = generator(prompt,
                        max_length=200)
    end = time.perf_counter()

    print(f"text generation:\n{outputs[0]['generated_text']}")
    print(f"Generator executed in {end - start} seconds using device: {generator.device}.\n")

def main():
    test_classifier()
    test_ner()
    test_question_answering()
    test_summarization()
    test_translation()
    test_text_generation()

if __name__ == '__main__':
    main()
