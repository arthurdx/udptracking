# UDP Tracking System

Sistema de rastreamento UDP com detecção de humanos usando YOLO e transmissão de vídeo em tempo real.

## Melhorias Implementadas

### Atomicidade e Ordem de Pacotes

Os scripts `udp_client.py` e `udp_server.py` foram refatorados para garantir:

1. **Protocolo Robusto**: Implementação de cabeçalhos estruturados com sequenciamento
2. **Atomicidade**: Envio e recepção de frames como unidades atômicas
3. **Ordem de Pacotes**: Controle de sequência para garantir ordem de entrega
4. **Buffer Adequado**: Configuração de buffers maiores para evitar erros 10040 no Windows

### Estrutura do Protocolo

```
Cabeçalho (20 bytes):
- frame_id (4 bytes): Identificador único do frame
- timestamp (8 bytes): Timestamp de envio
- size (4 bytes): Tamanho dos dados do frame
- sequence (4 bytes): Número de sequência para ordenação
```

### Características Técnicas

- **Tamanho Máximo de Pacote**: 65507 bytes (limite UDP)
- **Buffer de Recepção**: 65536 bytes
- **Fragmentação Automática**: Frames grandes são divididos em chunks
- **Controle de Sequência**: Verificação de ordem de pacotes
- **Logging Estruturado**: Logs em formato JSON com metadados completos

## Uso

### Servidor
```bash
python src/udp_server.py <video_file_path>
# ou para webcam:
python src/udp_server.py 0
```

### Cliente
```bash
python src/udp_client.py <ip_servidor> <porta>
```

## Funcionalidades

- **Detecção de Movimento**: Filtra frames sem movimento significativo
- **Detecção de Humanos**: Usa YOLO para detectar presença humana
- **Cooldown Inteligente**: Evita processamento excessivo após detecção
- **Monitoramento de Recursos**: CPU e memória em tempo real
- **Logs Detalhados**: Metadados completos de cada frame processado

## Estrutura de Arquivos

```
udptracking/
├── src/
│   ├── udp_client.py    # Cliente UDP refatorado
│   └── udp_server.py    # Servidor UDP refatorado
├── logs/                # Logs estruturados
├── videos/              # Vídeos salvos (opcional)
├── utils/
│   └── report.py        # Utilitários de relatório
└── requirements.txt     # Dependências
```

## Dependências

- OpenCV (cv2)
- Ultralytics (YOLO)
- NumPy
- psutil
- python-json-logger

## Solução para Erro 10040

O erro 10040 no Windows ocorria devido a:
- Buffer de recepção muito pequeno
- Falta de controle de tamanho de pacotes
- Ausência de fragmentação para frames grandes

**Solução implementada:**
- Buffers maiores (65536 bytes)
- Fragmentação automática de frames grandes
- Protocolo estruturado com cabeçalhos fixos
- Controle de sequência para ordenação 

## Logs Gerados

O sistema gera dois tipos de logs:

1. **`client_YYYYMMDD_HHMMSS.ndjson`**: Log do cliente com informações de recepção, latência e sequência
2. **`server_YYYYMMDD_HHMMSS.ndjson`**: Log do servidor com informações de processamento e detecção

**Nota**: O log do servidor é sobrescrito pelo cliente ao finalizar, garantindo que ambos os logs tenham o mesmo timestamp e sejam facilmente correlacionados. 