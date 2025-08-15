import os
import logging
from typing import List
import numpy as np # Necesitamos numpy para promediar
import vertexai
from vertexai.language_models import TextEmbeddingModel
from modules.config import config as settings

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

class ServicioVertexAI:
    def __init__(self):
        try:
            vertexai.init(project=settings.project_id, location=settings.location)
            logger.info(
                f"Vertex AI inicializado para el proyecto: {settings.project_id} en la región: {settings.location}"
            )
        except Exception as e:
            logger.warning(f"Vertex AI ya podría estar inicializado o hubo un warning: {e}")
            
        self.model_name = settings.embedding_model or "text-embedding-004"
        try:
            self.model = TextEmbeddingModel.from_pretrained(self.model_name)
            logger.info(f"Modelo de embeddings cargado: '{self.model_name}'")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo de embeddings '{self.model_name}': {e}")
            raise

    def generar_embedding_de_documento(self, documento: str, chunk_size: int = 2000, overlap: int = 200) -> List[float]:
        """
        Genera un único embedding para un documento largo dividiéndolo en chunks y promediando los resultados.

        Args:
            documento: El texto completo del documento.
            chunk_size: El tamaño de cada pedazo en caracteres.
            overlap: Cuántos caracteres se traslapan entre pedazos para mantener contexto.

        Returns:
            Un único vector de embedding que representa todo el documento.
        """
        if not isinstance(documento, str) or not documento.strip():
            logger.warning("Se recibió un documento vacío; retornando vector vacío.")
            return []

        logger.info(f"Iniciando la generación de embedding para un documento de {len(documento)} caracteres.")

        # 1. Dividir el documento en chunks
        chunks = []
        for i in range(0, len(documento), chunk_size - overlap):
            chunks.append(documento[i:i + chunk_size])
        
        logger.info(f"Documento dividido en {len(chunks)} chunks.")

        # 2. Obtener embeddings para cada chunk
        embeddings_de_chunks = self.get_text_embeddings(chunks)

        if not embeddings_de_chunks:
            logger.error("No se pudieron generar los embeddings para los chunks del documento.")
            return []

        # 3. Promediar los embeddings para obtener un vector final
        embedding_final = np.mean(embeddings_de_chunks, axis=0).tolist()
        
        logger.info(f"Embedding final del documento generado con dimensión: {len(embedding_final)}")
        return embedding_final


    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos, manejando los límites de la API de forma inteligente.
        """
        if not texts:
            return []

        # Limpiamos y validamos los textos de entrada
        textos_validos = [t for t in texts if isinstance(t, str) and t.strip()]
        if not textos_validos:
            return []

        all_embeddings: List[List[float]] = []
        
        # Límite de tokens por llamada (aproximado en caracteres para seguridad)
        # El modelo soporta 20000 tokens, lo dejamos en 18000 caracteres para tener margen.
        CHAR_LIMIT_PER_REQUEST = 18000 
        
        batch_actual = []
        chars_en_batch = 0
        
        logger.info(f"Procesando {len(textos_validos)} textos para generar embeddings...")

        for texto in textos_validos:
            # Truncamos textos individuales si son demasiado largos
            if len(texto) > CHAR_LIMIT_PER_REQUEST:
                logger.warning(f"Texto individual truncado de {len(texto)} a {CHAR_LIMIT_PER_REQUEST} caracteres.")
                texto = texto[:CHAR_LIMIT_PER_REQUEST]

            # Si agregar el siguiente texto excede el límite, procesamos el lote actual
            if chars_en_batch + len(texto) > CHAR_LIMIT_PER_REQUEST and batch_actual:
                logger.debug(f"Enviando lote de {len(batch_actual)} textos ({chars_en_batch} caracteres).")
                try:
                    embeddings = self.model.get_embeddings(batch_actual)
                    all_embeddings.extend([e.values for e in embeddings])
                except Exception as e:
                    logger.error(f"Error en lote de embeddings: {e}. Saltando este lote.")
                
                # Reseteamos el lote
                batch_actual = []
                chars_en_batch = 0

            # Agregamos el texto al lote actual
            batch_actual.append(texto)
            chars_en_batch += len(texto)

        # Procesamos el último lote que haya quedado
        if batch_actual:
            logger.debug(f"Enviando último lote de {len(batch_actual)} textos ({chars_en_batch} caracteres).")
            try:
                embeddings = self.model.get_embeddings(batch_actual)
                all_embeddings.extend([e.values for e in embeddings])
            except Exception as e:
                logger.error(f"Error en el último lote de embeddings: {e}.")

        logger.info(f"Embeddings generados exitosamente: {len(all_embeddings)} vectores.")
        return all_embeddings

