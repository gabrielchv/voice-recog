# 🎙️ Voz para Texto pt-BR

Transcrição de voz para texto em português brasileiro usando GPU NVIDIA (CUDA).
Leve o suficiente para rodar enquanto usa outros aplicativos.

## Requisitos

- NVIDIA GPU com 2GB+ VRAM (GTX 1650, etc.)
- Microfone funcionando
- Pop!_OS / Ubuntu / Linux (GNOME)

## Instalação

```bash
# 1. Instalar dependências do sistema (já feito)
sudo apt-get install portaudio19-dev

# 2. Ativar ambiente virtual
cd ~/Documents/voice_recog
source venv/bin/activate

# 3. Instalar pacotes (já feito)
pip install faster-whisper sounddevice pyperclip numpy scipy soundfile
```

O modelo Whisher base (~250MB) já foi baixado em `models/`.

## Como Usar

### Pelo Menu de Apps

Abra o menu de aplicativos e procure por **"Voz para Texto"**:

- **Voz para Texto pt-BR** → Abre um terminal com o app rodando em modo contínuo
- **Voz para Texto (Toggle)** → Liga/Desliga (pode ser associado a atalho de teclado)

### Pelo Terminal

```bash
cd ~/Documents/voice_recog

# Modo contínuo (escuta e transcreve automaticamente)
./voz-para-texto

# Gravar uma frase e transcrever
./voz-para-texto --oneshot

# Transcrever um arquivo de áudio
./voz-para-texto --file audio.wav

# Usar modelo small (melhor qualidade, mais VRAM)
./voz-para-texto --model small

# Forçar CPU (mais leve, mas mais lento)
./voz-para-texto --device cpu

# Listar dispositivos de áudio
./voz-para-texto --list-devices
```

### Atalho de Teclado (recomendado)

1. Abra **Configurações → Teclado → Atalhos → Personalizados**
2. Adicione:
   - **Nome:** Voz para Texto
   - **Comando:** `/home/gazerah/Documents/voice_recog/voz-toggle`
3. Atribua um atalho (ex: `Super+V`)

Agora `Super+V` liga/desliga a ferramenta!

### Toggle via terminal

```bash
./voz-toggle        # Liga se desligado, desliga se ligado
./voz-toggle on     # Liga
./voz-toggle off    # Desliga
./voz-toggle status # Mostra status
```

## Modelos Disponíveis

| Modelo | VRAM | Qualidade | Uso |
|--------|------|-----------|-----|
| tiny   | ~0.4GB | Básica | CPU / muita economia |
| base   | ~0.7GB | Boa    | **Recomendado** (padrão) |
| small  | ~1.2GB | Melhor | GPU com folga |
| medium | ~2.5GB | Ótima  | GPU 4GB+ |

Para trocar: `--model small` ou `--model tiny`

## Arquivos

| Arquivo | Função |
|---------|--------|
| `stt.py` | App principal |
| `voz-para-texto` | Lançador (ativa venv) |
| `voz-toggle` | Liga/desliga (para atalho) |
| `voz-para-texto.desktop` | Entry no menu de apps |
| `models/` | Modelos do Whisper baixados |
| `venv/` | Ambiente virtual Python |

## Dicas

- Fale em volume normal, a 20-30cm do microfone
- A transcrição é copiada automaticamente para a área de transferência
- Ctrl+C para parar o modo contínuo
- O modelo base consome ~700MB de VRAM - deixa espaço para outros apps
- Use `--model tiny` se quiser ainda mais economia
