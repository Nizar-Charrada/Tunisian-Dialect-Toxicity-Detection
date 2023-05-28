import torch
import os
import pyarabic.araby as araby
import pickle
from transliteration.models.model import TransformerConfig, TransformerModel
from toxic_detection.utils import ParamsNamespace, load_config
from toxic_detection.models.model import StudentModel
from transliteration.dataloader.preprocess import (
    preprocess_arabic,
    preprocess_arabizi,
    preprocess,
)
from transformers import AutoTokenizer, BertTokenizer
import torch.nn as nn


class Lang:
    def __init__(self, config, data, language):
        self.char2index = {}
        self.index2char = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.nchars = 3
        self.language = language
        self.is_target = language == config.target
        self.data = data

    def create_vocab(self):
        for word in self.data:
            self.addChar(word)

    def addChar(self, word):
        for char in word.lower():
            if char not in self.char2index:
                self.char2index[char] = self.nchars
                self.index2char[self.nchars] = char
                self.nchars += 1

    def tokenize(self, sentence, is_target=False):
        if is_target:
            return (
                [1] + [self.char2index[word] for word in sentence.lower()] + [2]
            )  # 1 is SOS and 2 is EOS
        else:
            return [self.char2index[word] for word in sentence.lower()]


def make_tokenizer(params):
    """
    Create a tokenizer based on the specified hyperparameters.
    """
    if params.model.language_model.model_type == "roberta":
        # here we used bert tokenizer for roberta model as stated in the model documentation "ziedsb19"
        tokenizer = BertTokenizer.from_pretrained(params.model.tokenizer.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(params.model.tokenizer.tokenizer_name)
    return tokenizer


def transliterate(text, model, source_vocab, target_vocab, transliteration_config):
    SOS_token = transliteration_config.SOS_token
    EOS_token = transliteration_config.EOS_token
    model.eval()
    input = (
        torch.LongTensor([source_vocab.char2index[i] for i in text])
        .unsqueeze(-1)
        .to(device)
    )
    preds = [SOS_token]

    while preds[-1] != EOS_token:
        tgt = torch.Tensor(preds).unsqueeze(-1).long().to(device)

        src = model.encoder_embd(input)
        src = model.pos_enc_embd(src)

        tgt = model.decoder_embd(tgt)
        tgt = model.pos_dec_embd(tgt)

        memory = model.transformer_encoder(src)
        output = model.transformer_decoder(tgt, memory)

        output = model.fc(output)

        pred = output.argmax(-1)[-1].item()

        preds.append(pred)

        if len(preds) > 50:
            """If the model is stuck in a loop, break."""
            preds = preds[: len(text)]
            break

    return "".join([target_vocab.index2char[i] for i in preds[1:-1]])


def extract_dialect_index(text):
    text_list = text.split(" ")
    idx = [0] * len(text_list)
    for i, word in enumerate(text_list):
        if araby.is_arabicrange(word):
            idx[i] = 1
        else:
            idx[i] = 0

    return text_list, idx


def extract_text(text):
    """Extract Arabic and Arabizi text from a sentence."""
    text_list, idx = extract_dialect_index(text)
    arabizi_script = [text for i, text in enumerate(text_list) if idx[i] == 0]
    arabic_script = [text for i, text in enumerate(text_list) if idx[i] == 1]
    return arabizi_script, arabic_script


def sentence_to_arabic(text):
    text_list, idx = extract_dialect_index(text)
    translation = []
    for i, t in enumerate(text_list):
        if idx[i] == 0:
            translation.append(
                transliterate(
                    t, arabizi_arabic_model, arabizi_vocab, arabic_vocab, trans_config
                )
            )
        else:
            translation.append(t)

    return " ".join(translation)


def sentence_to_arabizi(text):
    text_list, idx = extract_dialect_index(text)
    translation = []
    for i, t in enumerate(text_list):
        if idx[i] == 1:
            translation.append(
                transliterate(
                    t, arabic_arabizi_model, arabic_vocab, arabizi_vocab, trans_config
                )
            )
        else:
            translation.append(t)
    return " ".join(translation)


def predict(model, tokenizer, text):
    model.eval()
    tokens = tokenizer.tokenize(tokenizer.cls_token + text + tokenizer.sep_token)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)
    token_ids = token_ids.unsqueeze(0)

    pred = model(token_ids)
    logit = nn.Sigmoid()(pred)

    return logit


def multidialect_predict(text, threshold=0.5):
    text = preprocess(text)
    text = preprocess_arabic(text)
    text = preprocess_arabizi(text)

    arabizi_text, arabic_text = extract_text(text)
    arabizi_proba = 0
    arabic_proba = 0

    arabizi_text, arabic_text = " ".join(arabizi_text), " ".join(arabic_text)
    if len(arabizi_text):
        # arabizi only text
        arabizi_proba = predict(arabizi_model, arabizi_tokenizer, arabizi_text)

        # arabizi to arabic transliterated text
        arabizi_to_arabic_proba = predict(
            arabic_model, arabic_tokenizer, sentence_to_arabic(arabizi_text)
        )

        # full text transliteration to arabic
        mixed_tunbert_proba = predict(
            arabic_model, arabic_tokenizer, sentence_to_arabic(text)
        )
        print("Arabizi text only probability: %.2f" % arabizi_proba.item())
        print(
            "Arabizi to Arabic transliteration probability: %.2f"
            % arabizi_to_arabic_proba.item()
        )
        print(
            "Full text transliteration probability: %.2f" % mixed_tunbert_proba.item()
        )

    if len(arabic_text):
        # arabic only text
        arabic_proba = predict(arabic_model, arabic_tokenizer, arabic_text)

        # arabic to arabizi transliterated text
        print(sentence_to_arabizi(arabic_text))
        arabic_to_arabizi_proba = predict(
            arabizi_model, arabizi_tokenizer, sentence_to_arabizi(arabic_text)
        )

        # full text transliteration to arabizi
        mixed_zied_proba = predict(
            arabizi_model, arabizi_tokenizer, sentence_to_arabizi(text)
        )

        print("Arabic text only probability: %.2f" % arabic_proba.item())
        print(
            "Arabic to Arabizi transliteration probability: %.2f"
            % arabic_to_arabizi_proba.item()
        )
        print("Full text transliteration probability: %.2f" % mixed_zied_proba.item())


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------- Transliteration ----------------- #

    # we need to load the configuration file to get the source and target languages
    config_path = "transliteration\config\config.yaml"
    yaml_config = load_config(config_path)
    trans_config = ParamsNamespace(yaml_config)

    assert (
        trans_config.source == "arabizi" or trans_config.source == "arabic"
    ), "Source language not found, please choose arabizi or arabic"
    assert (
        trans_config.target == "arabizi" or trans_config.target == "arabic"
    ), "Target language not found, please choose arabizi or arabic"

    with open("transliteration/out/arabic_vocab.pkl", "rb") as f:
        arabic_vocab = pickle.load(f)
    with open("transliteration/out/arabizi_vocab.pkl", "rb") as f:
        arabizi_vocab = pickle.load(f)

    arabic_arabizi_args = TransformerConfig(
        **yaml_config["transformer"]
    )  # we need to load the model configuration

    arabizi_arabic_args = TransformerConfig(
        **yaml_config["transformer"]
    )  # we need to load the model configuration

    # we need to set the source and target vocabulary sizes
    arabic_arabizi_args.src_vocab_size = len(arabic_vocab.index2char)
    arabic_arabizi_args.tgt_vocab_size = len(arabizi_vocab.index2char)

    arabizi_arabic_args.src_vocab_size = len(arabizi_vocab.index2char)
    arabizi_arabic_args.tgt_vocab_size = len(arabic_vocab.index2char)

    arabizi_arabic_model = TransformerModel(
        arabizi_arabic_args,
    ).to(device)

    arabic_arabizi_model = TransformerModel(
        arabic_arabizi_args,
    ).to(device)

    checkpoint_path = os.path.join(trans_config.output_dir, f"arabic_to_arabizi.bin")
    arabic_arabizi_model.load_state_dict(torch.load(checkpoint_path))

    checkpoint_path = os.path.join(trans_config.output_dir, f"arabizi_to_arabic.bin")
    arabizi_arabic_model.load_state_dict(torch.load(checkpoint_path))

    # ----------------- Knowledge Distillation ----------------- #

    config_arabic_path = "toxic_detection/config/arabic_config.yaml"
    yaml_arabic_config = load_config(config_arabic_path)
    arabic_config = ParamsNamespace(yaml_arabic_config)

    config_arabizi_path = "toxic_detection/config/arabizi_config.yaml"
    yaml_arabizi_config = load_config(config_arabizi_path)
    arabizi_config = ParamsNamespace(yaml_arabizi_config)

    arabic_tokenizer = make_tokenizer(arabic_config)
    arabizi_tokenizer = make_tokenizer(arabizi_config)

    arabic_model = StudentModel(
        arabic_tokenizer.vocab_size,
        arabic_config.knowledge_distillation.student_model.embedding_size,
        arabic_config.knowledge_distillation.student_model.hidden_size,
        arabic_config.knowledge_distillation.student_model.kernel_size,
        arabic_config.knowledge_distillation.student_model.add_conv_layer,
    ).to(device)

    arabizi_model = StudentModel(
        arabizi_tokenizer.vocab_size,
        arabizi_config.knowledge_distillation.student_model.embedding_size,
        arabizi_config.knowledge_distillation.student_model.hidden_size,
        arabizi_config.knowledge_distillation.student_model.kernel_size,
        arabizi_config.knowledge_distillation.student_model.add_conv_layer,
    ).to(device)

    arabic_model.load_state_dict(
        torch.load(
            "toxic_detection/out/bert-knowledge_distillation-checkpoint/best.bin"
        )
    )
    arabizi_model.load_state_dict(
        torch.load(
            "toxic_detection/out/roberta-knowledge_distillation-checkpoint/best.bin"
        )
    )

    # ----------------- Toxic Detection ----------------- #

    while True:
        text = input("Enter a sentence: ")
        if text == "exit":
            break

        multidialect_predict(text)
