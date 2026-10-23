from typing import Collection, Iterable, Mapping, Sequence

from li_amber_input_provider import AmberInputsGenerative
from li_data_loader import DatasetJudge
from li_llm_messages import LlmAssistantMessage
from li_logging import TomlLogLike
from li_nltk_wrapper import CrossProcessNltkWrapper
import spacy
import spacy.tokens
import tomlkit

from li_amber_judge.amber_annotation_discriminative import AmberAnnotationDiscriminative
from li_amber_judge.amber_annotation_generative import AmberAnnotationGenerative
from li_amber_judge.amber_judgement_generative import AmberJudgementGenerative


class AmberJudgeGenerative(DatasetJudge[AmberJudgementGenerative]):
    # https://github.com/junyangwang0410/AMBER/blob/master/inference.py
    def __init__(self, 
                 annotations: Sequence[AmberAnnotationGenerative | AmberAnnotationDiscriminative], 
                 filter: AmberInputsGenerative,
                 association: Mapping[str, Sequence[str]], 
                 global_safe_words: set[str],
                 nltk: CrossProcessNltkWrapper,
                 spacy_encoreweblg: spacy.language.Language,
                 chair_threshold: float) -> None:
        super().__init__()

        self.__annotations: dict[str, tuple[Sequence[str], Sequence[str]]] = {}
        ids: set[str] = set(filter.ids)
        annotation: AmberAnnotationGenerative | AmberAnnotationDiscriminative
        for annotation in annotations:
            id: str = f"amber_{annotation["id"]}"
            if id in ids:
                assert annotation["type"] == "generative"
                assert id not in self.__annotations
                self.__annotations[id] = annotation["truth"], annotation["hallu"]

        self.__spacy_encoreweblg = spacy_encoreweblg
        self.__nltk = nltk

        self.__chair_threshold = chair_threshold

        self.__association = association
        self.__global_safe_words = global_safe_words
        
        self.__considered_words: set[str] = set()
        word: str
        for word in self.__association:
            self.__considered_words.add(word)
            for word in self.__association[word]:
                self.__considered_words.add(word)
    
    @property
    def ids(self) -> Iterable[str]:
        return self.__annotations.keys()

    def __are_similar(self, word1: str, word2: str, similarity_score: float = 0.8):
        if word1 == word2:
            return True
        
        token1: spacy.tokens.Doc = self.__spacy_encoreweblg(word1)
        token2: spacy.tokens.Doc = self.__spacy_encoreweblg(word2)
        similarity: float = token1.similarity(token2)
        return similarity > similarity_score
    
    def __len_divide(self, left: Collection, right: Collection, if_zero: float = 0.):
        right_len: int = len(right)
        return (len(left) / right_len) if right_len != 0 else if_zero

    def __extract_nouns(self, text: str) -> list[str]:
        tokens: list[str] = self.__nltk.word_tokenize(text)
        tagged: list[tuple[str, str]] = self.__nltk.pos_tag(tokens)
        return self.__nltk.word_net_lemmatizer_lemmatize_many([
            (word, "n") 
            for word, pos in tagged 
            if pos.startswith("NN")
        ])
    
    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> AmberJudgementGenerative:
        with log:
            log.set_document_value(id, "data_id")
            log.set_document_value(self.__chair_threshold, "chair_threshold")

            annotation_truth: Sequence[str]
            annotation_hallu: Sequence[str]
            annotation_truth, annotation_hallu = self.__annotations[id]

            word_to_truth: dict[str, str] = {}
            truth_word: str
            for truth_word in annotation_truth:
                word_to_truth[truth_word] = truth_word
                word: str
                for word in self.__association[truth_word]:
                    word_to_truth[word] = truth_word
            log.set_document_value(annotation_truth, "truth")
            log.set_document_value(word_to_truth, "word_to_truth")

            word_to_hallu: dict[str, str] = {}
            hallu_word: str
            for hallu_word in annotation_hallu:
                word_to_hallu[hallu_word] = hallu_word
                for word in self.__association[hallu_word]:
                    word_to_hallu[word] = hallu_word
            log.set_document_value(annotation_hallu, "hallu")
            log.set_document_value(word_to_hallu, "word_to_hallu")

            log.set_document_value(response.content, "raw_response")

            
            nouns: list[str] = self.__extract_nouns(response.content)
            log.set_document_value(nouns, "nouns")
            
            considered: list[str] = []
            found_truth: set[str] = set()
            found_hallu: set[str] = set()
            non_truth: list[str] = []

            analysis: list[tuple[str, list[str]]] = []

            for word in nouns:
                if word not in self.__considered_words:
                    analysis.append((word, ["Is not in considered words"]))
                    continue

                considered.append(word)

                if word in self.__global_safe_words:
                    analysis.append((word, ["Is in global safe words"]))
                    continue
                
                current_analysis: list[str] = []
                analysis.append((word, current_analysis))

                is_truth: bool = False
                for truth_word in word_to_truth:
                    if self.__are_similar(word, truth_word):
                        current_analysis.append(f"Is similar to truth '{truth_word}'")
                        found_truth.add(word_to_truth[truth_word])
                        is_truth = True

                if not is_truth:
                    current_analysis.append(f"Is not similar to any truth")
                    non_truth.append(word)

                for hallu_word in word_to_hallu:
                    if self.__are_similar(word, hallu_word):
                        current_analysis.append(f"Is similar to hallu '{hallu_word}'")
                        found_hallu.add(word_to_hallu[hallu_word])
            
            log.set_document_value([[a[0], tomlkit.item(a[1]).multiline(True)] for a in analysis], "analysis")

            log.set_document_value(considered, "all_considered_nouns")
            log.set_document_value(non_truth, "non_truth")
            log.set_document_value(found_truth, "found_truth")
            log.set_document_value(found_hallu, "found_hallu")

            truth_cover_score: float = self.__len_divide(found_truth, annotation_truth)
            log.set_document_value(truth_cover_score, "truth_cover_score")
            hallu_cover_score: float = self.__len_divide(found_hallu, annotation_hallu)
            log.set_document_value(hallu_cover_score, "hallu_cover_score")
            chair: float = self.__len_divide(non_truth, considered)
            log.set_document_value(chair, "chair_score")

            result: bool = True
            reason: str = ""
            if chair > self.__chair_threshold:
                result = False
                reason = f"The CHAIR score '{chair}' is higher than the threshold '{self.__chair_threshold}'."
            if len(found_hallu) != 0:
                if result == True:
                    result = False
                    reason += f"Some hallucination objects are mentioned in the answer: '{found_hallu}."
                else:
                    reason += f" In addition, some hallucination objects are mentioned in the answer: '{found_hallu}."

            log.set_document_value(result, "result")
            log.set_document_value(reason, "reason")
            return AmberJudgementGenerative(result, reason,
                                            found_truth, annotation_truth,
                                            found_hallu, annotation_hallu,
                                            non_truth, considered,
                                            truth_cover_score, hallu_cover_score,
                                            chair)
        