"""Implements the co-reference matcher"""

from abc import ABC, abstractmethod
from itertools import chain

from inflection import pluralize
from spacy.matcher import Matcher


class AbstractCoreferenceRecognizer(ABC):

    """Implements the coreference recognizer"""

    _SPACE = " "

    def __init__(self):
        self._presumptive_reference_text = None
        self._utterance = None

    def presumptive_reference_text(self):
        """Returns the presumptive dialog text"""
        return self._presumptive_reference_text

    @abstractmethod
    def recognize(self, coreference_context, utterance):
        """Recognizes the coreferences"""

    @abstractmethod
    def resolve(self):
        """Resolves the recognized coreferences"""


class AbstractEnglishCoreferenceRecognizer(AbstractCoreferenceRecognizer):

    """Implements the English coreference recognizer"""

    _ITEM_SEPERATOR = " or "
    _PARENTHESES = ["(", ")"]
    _PRESUMPTIVE_SINGULAR_START_PHRASE = "There is a"
    _SENTENCE_SEPERATOR = ". "
    _WITH_PREPOSITION = "with"

    def __init__(self, spacy_model):
        super().__init__()
        self._spacy_model = spacy_model
        self._utterance_doc = None

    @staticmethod
    def _item_patterns(item_name):
        """Returns the item partterns for the Spacy Matcher"""
        item_pattern = [{"LEMMA": "-PRON-"}]
        item_pattern.extend(
            [{"LEMMA": item_token} for item_token in item_name.split(AbstractCoreferenceRecognizer._SPACE)])
        return [item_pattern]

    @staticmethod
    def _prepare_dialog_item_from_filter_phrase(dialog_items, pluralized_entity_name, plural_query_context):
        """Prepares the query context item from filter phrase"""
        filter_phrase = plural_query_context.get_filterPhrase()
        if not filter_phrase:
            dialog_items.append(pluralized_entity_name)
            return
        dialog_items.append(
            AbstractCoreferenceRecognizer._SPACE.join(
                [pluralized_entity_name, filter_phrase.join(AbstractEnglishCoreferenceRecognizer._PARENTHESES)]))

    def _prepare_dialog_items_from_query_context_items(
            self, attribute_reference, dialog_items, entity_name, query_context, presumptive):
        """Prepares the dialog items from the query context items"""
        def query_context_items_phrase(item_name, item_values):
            item_phrases = []
            for item_value in item_values:
                query_context_item_phrase = AbstractCoreferenceRecognizer._SPACE.join(
                    [item_name, item_value])
                item_phrase_tokens = [
                    AbstractEnglishCoreferenceRecognizer._WITH_PREPOSITION, query_context_item_phrase] if \
                    not attribute_reference else [query_context_item_phrase]
                item_phrases.append(AbstractCoreferenceRecognizer._SPACE.join(item_phrase_tokens))
            if len(item_values) == 1:
                tokens = [
                    AbstractEnglishCoreferenceRecognizer._PRESUMPTIVE_SINGULAR_START_PHRASE,
                    AbstractEnglishCoreferenceRecognizer._ITEM_SEPERATOR.join(item_phrases)] if presumptive else \
                    [AbstractEnglishCoreferenceRecognizer._ITEM_SEPERATOR.join(item_phrases)]
                if attribute_reference:
                    return AbstractCoreferenceRecognizer._SPACE.join(tokens)
                return entity_name + AbstractCoreferenceRecognizer._SPACE + AbstractCoreferenceRecognizer._SPACE.join(
                    tokens)
            if attribute_reference:
                return AbstractEnglishCoreferenceRecognizer._ITEM_SEPERATOR.join(item_phrases)
            return entity_name + AbstractCoreferenceRecognizer._SPACE + \
                AbstractEnglishCoreferenceRecognizer._ITEM_SEPERATOR.join(item_phrases)

        items_matched = False
        for query_context_item in query_context.get_items():
            matcher = Matcher(self._spacy_model.vocab)
            item_name = query_context_item.get_name()
            for pattern in AbstractEnglishCoreferenceRecognizer._item_patterns(item_name):
                matcher.add(item_name, None, pattern)
            matches = matcher(self._utterance_doc)
            if not matches:
                continue
            items_matched |= True
            dialog_items.append(query_context_items_phrase(item_name, query_context_item.get_values()))
        if items_matched:
            return
        for query_context_item in query_context.get_primaryKeyItems():
            dialog_items.append(
                query_context_items_phrase(query_context_item.get_name(), query_context_item.get_values()))

    def _prepare_plural_query_context_phrase(
            self, dialog_items, plural_query_context, attribute_reference=False, presumptive=False):
        """Prepares the plural query context phrase"""
        if not plural_query_context:
            return
        pluralized_entity_name = pluralize(plural_query_context.get_entityName())
        self._prepare_dialog_items_from_query_context_items(
            attribute_reference, dialog_items, pluralized_entity_name, plural_query_context, presumptive=presumptive)
        AbstractEnglishCoreferenceRecognizer._prepare_dialog_item_from_filter_phrase(
            dialog_items, pluralized_entity_name, plural_query_context)

    def _prepare_singular_query_context_phrase(
            self, dialog_items, singular_query_context, attribute_reference=False, presumptive=False):
        """Prepares the singular query context phrase"""
        if not singular_query_context:
            return
        entity_name = singular_query_context.get_entityName()
        self._prepare_dialog_items_from_query_context_items(
            attribute_reference, dialog_items, entity_name, singular_query_context, presumptive=presumptive)

    @abstractmethod
    def recognize(self, coreference_context, utterance):
        """Recognizes the coreferences"""

    @abstractmethod
    def resolve(self):
        """Resolves the recognized coreferences"""


class IndonesianCoreferenceRecognizer(AbstractCoreferenceRecognizer):

    """Implements the coreference recognizer for the Indonesian language"""

    def recognize(self, coreference_context, utterance):
        self._utterance = utterance

    def resolve(self):
        return self._utterance

class MultiLanguageCoreferenceRecognizer(AbstractCoreferenceRecognizer):

    """Implements the multi language coreference recognizer"""

    def __init__(self):
        super().__init__()
        self._coreference_recognizers = []

    def add_recognizer(self, coreference_recognizer):
        """Adds the coreference_recognizer"""
        self._coreference_recognizers.append(coreference_recognizer)

    def recognize(self, coreference_context, utterance):
        self._utterance = utterance
        if not coreference_context:
            return
        presumptive_reference_texts = []
        for coreference_recognizer in self._coreference_recognizers:
            coreference_recognizer.recognize(coreference_context, utterance)
            utterance = coreference_recognizer.resolve()
            self._utterance = coreference_recognizer.resolve()
            presumptive_reference_text = coreference_recognizer.presumptive_reference_text()
            if presumptive_reference_text:
                presumptive_reference_texts.append(presumptive_reference_text)
        self._presumptive_reference_text = AbstractCoreferenceRecognizer._SPACE.join(presumptive_reference_texts)

    def resolve(self):
        return self._utterance


class RuleBasedEnglishCoreferenceRecognizer(AbstractEnglishCoreferenceRecognizer):

    """Implements a rule based English language coreference recognizer"""

    _COMMON_OBJECT_PRONOUNS = {}
    _COMMON_PERSON_PRONOUNS = {"you", "your", "yours"}
    _COMMON_PLURAL_PRONOUNS = {"they", "their", "them", "theirs"}
    _COMMON_SINGULAR_PRONOUNS = {"it", "its"}
    _PLURAL_OBJECT_PRONOUNS = {}
    _PLURAL_PERSON_PRONOUNS = {"our", "ours", "we", "us"}
    _PLURAL_PRONOUNS = set(chain(_COMMON_PLURAL_PRONOUNS, _PLURAL_OBJECT_PRONOUNS, _PLURAL_PERSON_PRONOUNS))
    _PRONOUN_PATTERN = [{"LEMMA": "-PRON-"}]
    _PRONOUN_PATTERN_NAME = "pronoun"
    _SINGULAR_OBJECT_PRONOUNS = {}
    _SINGULAR_PERSON_PRONOUNS = {"i", "me", "mine", "he", "him", "his", "she", "her", "hers"}
    _SINGULAR_PRONOUNS = set(chain(_COMMON_SINGULAR_PRONOUNS, _SINGULAR_OBJECT_PRONOUNS, _SINGULAR_PERSON_PRONOUNS))

    def __init__(self, spacy_model):
        super().__init__(spacy_model)
        self._pronoun_matcher = Matcher(self._spacy_model.vocab)
        self._pronoun_matcher.add(
            RuleBasedEnglishCoreferenceRecognizer._PRONOUN_PATTERN_NAME, None,
            RuleBasedEnglishCoreferenceRecognizer._PRONOUN_PATTERN)
        self._coreference_context = None

    @staticmethod
    def _item_names(query_context):
        """Returns the item names"""
        if not query_context:
            return None
        items = query_context.get_items()
        if not items:
            return None
        return {item.get_name() for item in items}

    def _match_item_patterns(self):
        """Matches the item pattern"""
        plural_object_item_names = \
            RuleBasedEnglishCoreferenceRecognizer._item_names(self._coreference_context.get_pluralObjectQueryContext())
        plural_person_item_names = \
            RuleBasedEnglishCoreferenceRecognizer._item_names(self._coreference_context.get_pluralPersonQueryContext())
        singular_object_item_names = \
            RuleBasedEnglishCoreferenceRecognizer._item_names(
                self._coreference_context.get_singularObjectQueryContext())
        singular_person_query_context = \
            RuleBasedEnglishCoreferenceRecognizer._item_names(
                self._coreference_context.get_singularPersonQueryContext())
        matcher = Matcher(self._spacy_model.vocab)
        for item_name in chain(
            *(
                item_names for item_names in (
                    plural_object_item_names, plural_person_item_names, singular_object_item_names,
                    singular_person_query_context) if item_names)):
            for pattern in AbstractEnglishCoreferenceRecognizer._item_patterns(item_name):
                matcher.add(item_name, None, pattern)
        matches = matcher(self._utterance_doc)
        if not matches:
            return []
        matched_texts = []
        for _, start_idx, end_idx in matches:
            matched_texts.append(self._utterance_doc[start_idx:end_idx].text)
        return matched_texts

    def _match_pronoun(self, text):
        """Matches the text with matcher"""
        text_doc = self._spacy_model(text)
        matches = self._pronoun_matcher(text_doc)
        if not matches:
            return None
        _, start_idx, end_idx = matches[0]
        return text_doc[start_idx:end_idx].text

    def _match_pronouns(self):
        """Matches the utterance with matcher"""
        matches = self._pronoun_matcher(self._utterance_doc)
        if not matches:
            return []
        matched_texts = []
        for _, start_idx, end_idx in matches:
            matched_texts.append(self._utterance_doc[start_idx:end_idx].text)
        return matched_texts

    def _recognize_attribute_references(self):
        """Recognizes the attribute name references in the utterance"""
        for matched_text in self._match_item_patterns():
            pronoun = self._match_pronoun(matched_text)
            self._recognize_common_pronouns(attribute_reference=True, matched_text=matched_text, pronoun=pronoun)

    def _recognize_common_pronouns(self, matched_text, pronoun, attribute_reference=False):
        """Recognizes the common pronouns"""
        self._recognize_singular_pronouns(matched_text, pronoun, attribute_reference=attribute_reference)
        self._recognize_plural_pronouns(matched_text, pronoun, attribute_reference=attribute_reference)

    def _recognize_entity_references(self):
        """Recognizes the entity references"""
        for pronoun in self._match_pronouns():
            self._recognize_common_pronouns(matched_text=pronoun, pronoun=pronoun)

    def _resolve_pronouns(self, attribute_reference, context_resolution_function, coreference_context, matched_text):
        """Resolves the pronouns"""
        dialog_items = []
        context_resolution_function(dialog_items, coreference_context, attribute_reference)
        self._resolve_query_context(dialog_items, matched_text)

    def _recognize_plural_pronouns(self, matched_text, pronoun, attribute_reference=False):
        """Recognizes the plural pronouns"""
        if pronoun not in RuleBasedEnglishCoreferenceRecognizer._PLURAL_PRONOUNS:
            return
        if pronoun not in RuleBasedEnglishCoreferenceRecognizer._PLURAL_PERSON_PRONOUNS:
            context = self._coreference_context.get_pluralObjectQueryContext()
            self._resolve_pronouns(
                attribute_reference, self._prepare_plural_query_context_phrase, context, matched_text)
        if pronoun not in RuleBasedEnglishCoreferenceRecognizer._PLURAL_OBJECT_PRONOUNS:
            context = self._coreference_context.get_pluralPersonQueryContext()
            self._resolve_pronouns(
                attribute_reference, self._prepare_plural_query_context_phrase, context, matched_text)

    def _recognize_singular_pronouns(self, matched_text, pronoun, attribute_reference=False):
        """Recognizes the singular pronouns"""
        if pronoun not in RuleBasedEnglishCoreferenceRecognizer._SINGULAR_PRONOUNS:
            return
        if pronoun not in RuleBasedEnglishCoreferenceRecognizer._SINGULAR_PERSON_PRONOUNS:
            self._resolve_pronouns(
                attribute_reference, self._prepare_singular_query_context_phrase,
                self._coreference_context.get_singularObjectQueryContext(), matched_text)
        if pronoun not in RuleBasedEnglishCoreferenceRecognizer._SINGULAR_OBJECT_PRONOUNS:
            self._resolve_pronouns(
                attribute_reference, self._prepare_singular_query_context_phrase,
                self._coreference_context.get_singularPersonQueryContext(), matched_text)

    def _resolve_query_context(self, dialog_items, matched_text):
        """Resolves the query context"""
        if not dialog_items:
            return
        self._utterance = self._utterance.replace(matched_text, dialog_items[0])

    def recognize(self, coreference_context, utterance):
        self._utterance = utterance
        if not coreference_context:
            return
        self._utterance_doc = self._spacy_model(self._utterance)
        self._coreference_context = coreference_context
        self._recognize_attribute_references()
        self._recognize_entity_references()

    def resolve(self):
        return self._utterance
