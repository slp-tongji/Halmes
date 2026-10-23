import json
from collections import defaultdict
from pathlib import Path
import threading
from typing import Iterable, Sequence, TypedDict

from frozendict import frozendict
from li_nltk_wrapper import CrossProcessNltkWrapper

import csfile
from li_data_loader import DatasetJudge
from li_llm_messages import LlmAssistantMessage
from li_logging import TomlLogLike

from li_coco_judge.coco_judgement import CocoJudgement
from li_coco_judge.coco_extension_for_chair import CocoExtensionForChair


class CocoJudge(DatasetJudge[CocoJudgement]):
    # https://github.com/shikiw/OPERA/blob/main/chair.py

    class _Image(TypedDict):
        id: int
        # there are some other fields, but they won't be used.

    class _InstancesAnnotation(TypedDict):
        image_id: int
        category_id: int
        # there are some other fields, but they won't be used.

    class _InstancesCategory(TypedDict):
        id: int
        name: str
        # there are some other fields, but they won't be used.

    class _InstancesFile(TypedDict):
        images: Sequence['CocoJudge._Image']
        annotations: Sequence['CocoJudge._InstancesAnnotation']
        categories: Sequence['CocoJudge._InstancesCategory']
        # there are some other fields, but they won't be used.

    class _CaptionsAnnotation(TypedDict):
        image_id: int
        caption: str
        # there are some other fields, but they won't be used.

    class _CaptionsFile(TypedDict):
        images: Sequence['CocoJudge._Image']
        annotations: Sequence['CocoJudge._CaptionsAnnotation']
        # there are some other fields, but they won't be used.

    def __init__(
        self,
        captions_file: Path,
        instances_file: Path,
        nltk: CrossProcessNltkWrapper,
        chair_threshold: float = 0.,
    ) -> None:
        super().__init__()

        instances: CocoJudge._InstancesFile = json.loads(csfile.read_all_text(instances_file))

        self.__nltk: CrossProcessNltkWrapper = nltk
        self.__chair_threshold = chair_threshold
        self.__data: dict[str, set[str]] = defaultdict(set)
        
        categories: dict[int, str] = {c["id"]: c["name"] for c in instances["categories"]}
        synonyms: frozendict[str, str] = CocoExtensionForChair.synonyms()
        annotation_instances: CocoJudge._InstancesAnnotation
        for annotation_instances in instances["annotations"]:
            node: str | None = synonyms.get(categories[annotation_instances["category_id"]])
            if node is not None:
                self.__data[f"coco_{annotation_instances["image_id"]}"].add(node)

        self.__captions: dict[str, str | None] = {}
        captions: CocoJudge._CaptionsFile = json.loads(csfile.read_all_text(captions_file))
        annotation_captions: CocoJudge._CaptionsAnnotation
        for annotation_captions in captions["annotations"]:
            self.__captions[f"coco_{annotation_captions["image_id"]}"] = annotation_captions["caption"]

        self.__data_update_lock = threading.Lock()

    def __get_data(self, id: str):
        caption: str | None = self.__captions[id]
        if caption is not None:
            with self.__data_update_lock:
                if caption is not None:
                    words: list[str] = [w for w, _ in self.__caption_to_words(caption)]
                    self.__data[id].update(words)
        return self.__data[id]
        
    @staticmethod
    def __get_wordnet_pos(tag: str):
        if tag.startswith("J"):
            return "a"
        if tag.startswith("V"):
            return "v"
        if tag.startswith("N"):
            return "n"
        if tag.startswith("R"):
            return "r"
        return None

    def __word_tokenize(self, caption: str):
        caption = caption.lower()
        words: list[str] = self.__nltk.word_tokenize(caption)

        start: int = 0
        word: str
        for word in words:
            start = caption.find(word, start)
            yield word, start
            start += len(word)

    def __caption_to_words(self, caption: str) -> list[tuple[str, int]]:
        words: list[tuple[str, int]] = list(self.__word_tokenize(caption))

        words = list(zip(
            self.__nltk.word_net_lemmatizer_lemmatize_many([
                (word, self.__get_wordnet_pos(tag) or "n")
                for word, tag in self.__nltk.pos_tag([w for w, _ in words])
            ]), 
            (i for _, i in words)
        ))

        double_words: frozendict[str, str] = CocoExtensionForChair.double_words()

        i: int = 0
        merged: list[tuple[str, int]] = []
        while i < len(words):
            next_two: str = " ".join((w for w, _ in words[i:(i + 2)]))
            double_word: str | None = double_words.get(next_two)
            if double_word is not None:
                merged.append((double_word, words[i][1]))
                i += 2
                continue
            merged.append(words[i])
            i += 1
        words = merged

        if "toilet" in words and "seat" in words:
            words = [w for w in words if w[0] != "seat"]

        synonyms: frozendict[str, str] = CocoExtensionForChair.synonyms()
        
        result: list[tuple[str, int]] = []
        word: str | None
        index: int
        for word, index in words:
            word = synonyms.get(word)
            if word is not None:
                result.append((word, index))
        return result

    @property
    def ids(self) -> Iterable[str]:
        return self.__data.keys()

    def judge(self, id: str, response: LlmAssistantMessage, log: TomlLogLike) -> CocoJudgement:
        with log:
            log.set_document_value(id, "data_id")
            true_objects: set[str] = self.__get_data(id)
            log.set_document_value(true_objects, "true_objects")
            log.set_document_value(self.__chair_threshold, "chair_threshold")

            log.set_document_value(response.content, "response")

            response_words: list[tuple[str, int]] = self.__caption_to_words(response.content)

            hallucination_count: int = 0
            tagged_response_words: list[tuple[str, int, bool]] = []
            word: str
            index: int
            for word, index in response_words:
                if word in true_objects:
                    tagged_response_words.append((word, index, True))
                else:
                    tagged_response_words.append((word, index, False))
                    hallucination_count += 1

            log.set_document_value(tagged_response_words, "tagged_response_words")

            chair_score: float = (
                hallucination_count / len(tagged_response_words) 
                if len(response_words) != 0 else 0.0
            )

            result: bool
            reason: str
            if chair_score > self.__chair_threshold:
                result = False
                reason = f"The CHAIR score '{chair_score}' is higher than the threshold '{self.__chair_threshold}'."
            else:
                result = True
                reason = ""

            log.set_document_value(result, "result")
            log.set_document_value(reason, "reason")

            return CocoJudgement(result, reason,
                                 response.content, 
                                 tagged_response_words,
                                 chair_score,
                                 true_objects)
    