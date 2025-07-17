import numpy as np
from spacy.language import Language

@Language.factory('tensor2attr')
class Tensor2Attr:
    """SpaCy pipeline component to add tensor attributes to tokens"""
    def __init__(self, name, nlp):
        pass

    def __call__(self, doc):
        self.add_attributes(doc)
        return doc

    def add_attributes(self, doc):
        """Add vector and similarity attributes to document, spans, and tokens"""
        doc.user_hooks['vector'] = self.doc_tensor
        doc.user_span_hooks['vector'] = self.span_tensor
        doc.user_token_hooks['vector'] = self.token_tensor
        
        doc.user_hooks['similarity'] = self.get_similarity
        doc.user_span_hooks['similarity'] = self.get_similarity
        doc.user_token_hooks['similarity'] = self.get_similarity

    def doc_tensor(self, doc):
        """Get the tensor for a whole document"""
        return doc._.trf_data.last_hidden_layer_state.data.mean(axis=0)

    def span_tensor(self, span):
        """Get the tensor for a span by averaging its token tensors"""
        tensors = []
        for i in range(span.start, span.end):
            tensors.append(self.token_tensor(span.doc[i]))
        return np.mean(tensors, axis=0) if tensors else np.zeros(768)

    def token_tensor(self, token):
        """Get the contextual embedding for a token from the transformer"""
        if not hasattr(token.doc._, 'trf_data'):
            return token.vector  # Fallback to static vector
        
        token_tensor = token.doc._.trf_data.last_hidden_layer_state[token.i].data
        # Average if there are multiple subword pieces
        return token_tensor.mean(axis=0) if token_tensor.size > 0 else token.vector

    def get_similarity(self, doc1, doc2):
        """Compute cosine similarity between two documents"""
        return np.dot(doc1.vector, doc2.vector) / (doc1.vector_norm * doc2.vector_norm)