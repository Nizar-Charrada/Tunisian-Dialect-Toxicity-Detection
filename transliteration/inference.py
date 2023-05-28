import torch
import pickle
from  models.model import TransformerModel
import os
import yaml
from models.model import TransformerConfig
from utils import ParamsNamespace, detect_arabic_type
from train import Lang
from dataloader.preprocess import preprocess_arabic, preprocess_arabizi


def transliterate(text):
    """Transliterate a sentence from source to target language
    Args:
        text (str): sentence to transliterate"""

    model.eval()

    input = (
        torch.LongTensor([source_vocab.char2index[i] for i in text])
        .unsqueeze(-1)
        .to(device)
    )

    preds = [config.SOS_token]

    # the model will keep predicting until it predicts an EOS token
    while preds[-1] != config.EOS_token:
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

        # to avoid infinite loop
        if len(preds) > 50:
            preds = preds[: len(text)]
            break

    return "".join([target_vocab.index2char[i] for i in preds[1:-1]])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # we need to load the configuration file to get the source and target languages
    config_path = "transliteration\config\config.yaml"

    assert os.path.isfile(config_path), "No configuration file found at {}".format(
        config_path
    )
    with open(config_path) as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)

    config = ParamsNamespace(yaml_config)

    assert (
        config.source == "arabizi" or config.source == "arabic"
    ), "Source language not found, please choose arabizi or arabic"
    assert (
        config.target == "arabizi" or config.target == "arabic"
    ), "Target language not found, please choose arabizi or arabic"
    with open(f"transliteration\out\{config.source}_vocab.pkl", "rb") as f:
        source_vocab = pickle.load(f)
    with open(f"transliteration/out/{config.target}_vocab.pkl", "rb") as f:
        target_vocab = pickle.load(f)

    model_args = TransformerConfig(
        **yaml_config["transformer"]
    )  # we need to load the model configuration

    # we need to set the source and target vocabulary sizes
    model_args.src_vocab_size = len(source_vocab.index2char)
    model_args.tgt_vocab_size = len(target_vocab.index2char)

    model = TransformerModel(
        model_args,
    ).to(device)

    checkpoint_path = os.path.join(
        config.output_dir, f"{source_vocab.language}_to_{target_vocab.language}.bin"
    )

    model.load_state_dict(torch.load(checkpoint_path))

    # -------------------------------------------------------------------------------------------------------------
    # The text should be in the source language (arabizi or arabic) we wont check for that here for simplicity sake

    text = input("Enter the text to transliterate: ")  # we get the text from the user

    # we need to preprocess the text before feeding it to the model
    if detect_arabic_type(text) == "Arabic" and config.source == "arabic":
        preprocessed_text = preprocess_arabic(text)
    elif detect_arabic_type(text) == "Arabizi" and config.source == "arabizi":
        preprocessed_text = preprocess_arabizi(text)
    else:
        assert False, f"The text is not in the {config.source} language"

    transliterated = transliterate(preprocessed_text)
    print("Original text: ", text)
    print("Preprocessed text: ", preprocessed_text)
    print("Transliterated text: ", transliterated)
