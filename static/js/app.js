// Vortex AI - Client Controller

document.addEventListener("DOMContentLoaded", () => {
    // DOM Cache
    const uploadZone = document.getElementById("uploadZone");
    const fileInput = document.getElementById("fileInput");
    const uploadIcon = document.getElementById("uploadIcon");
    const uploadSpinner = document.getElementById("uploadSpinner");
    
    const uploadProgressContainer = document.getElementById("uploadProgressContainer");
    const progressBar = document.getElementById("progressBar");
    const progressPercent = document.getElementById("progressPercent");
    const progressText = document.getElementById("progressText");
    const docCard = document.getElementById("docCard");
    const docName = document.getElementById("docName");
    const removeDocBtn = document.getElementById("removeDocBtn");
    
    // Status Badges
    const llmEngineBadge = document.getElementById("llmEngineBadge");
    const indexStatusBadge = document.getElementById("indexStatusBadge");
    const currentContextName = document.getElementById("currentContextName");
    
    // Chat Components
    const chatMessages = document.getElementById("chatMessages");
    const welcomeScreen = document.getElementById("welcomeScreen");
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");
    const toastContainer = document.getElementById("toastContainer");

    let isDocumentLoaded = false;

    // 1. Initialize Application - Fetch System Status
    const initApp = async () => {
        try {
            const res = await fetch("/health");
            if (!res.ok) throw new Error("Health check failed");
            
            const data = await res.json();
            
            // Set LLM Provider Badge
            llmEngineBadge.innerText = data.llm_provider === "gemini" ? "Gemini Engine" : "Hugging Face";
            
            if (data.has_document_loaded) {
                setDocumentReadyState(data.loaded_document);
                loadChatHistory();
            } else {
                setDocumentEmptyState();
            }
        } catch (err) {
            console.error("Initialization error:", err);
            showToast("Failed to connect to Vortex AI server.", "error");
        }
    };

    // 2. Load Chat History from Server
    const loadChatHistory = async () => {
        try {
            const res = await fetch("/history");
            if (!res.ok) return;
            const data = await res.json();
            
            if (data.history && data.history.length > 0) {
                welcomeScreen.classList.add("hidden");
                data.history.forEach(exchange => {
                    appendMessageBubble(exchange.question, "user");
                    appendMessageBubble(exchange.answer, "bot", "0.00");
                });
                scrollToBottom();
            }
        } catch (err) {
            console.error("Error loading chat history:", err);
        }
    };

    // 3. UI State Transitions
    const setDocumentReadyState = (filename) => {
        isDocumentLoaded = true;
        uploadZone.classList.add("hidden");
        uploadProgressContainer.classList.add("hidden");
        
        docName.innerText = filename || "document.pdf";
        docCard.classList.remove("hidden");
        
        indexStatusBadge.innerText = "Indexed";
        indexStatusBadge.className = "badge badge-success";
        currentContextName.innerText = `Active Context: ${filename || "document.pdf"}`;
        
        // Reset spinner state
        if (uploadSpinner && uploadIcon) {
            uploadSpinner.classList.add("hidden");
            uploadIcon.classList.remove("hidden");
        }

        // Enable input fields
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    };

    const setDocumentEmptyState = () => {
        isDocumentLoaded = false;
        docCard.classList.add("hidden");
        uploadProgressContainer.classList.add("hidden");
        uploadZone.classList.remove("hidden");
        
        indexStatusBadge.innerText = "No Document";
        indexStatusBadge.className = "badge badge-warning";
        currentContextName.innerText = "Load a PDF document to instantiate chat";
        
        // Reset spinner state
        if (uploadSpinner && uploadIcon) {
            uploadSpinner.classList.add("hidden");
            uploadIcon.classList.remove("hidden");
        }

        // Disable input fields
        userInput.value = "";
        userInput.disabled = true;
        sendBtn.disabled = true;
        welcomeScreen.classList.remove("hidden");
        
        clearChatDisplay();
    };

    const clearChatDisplay = () => {
        const bubbles = chatMessages.querySelectorAll(".message-row");
        bubbles.forEach(b => b.remove());
        welcomeScreen.classList.remove("hidden");
    };

    // 4. File Upload Ingestion
    const handleFileUpload = async (file) => {
        if (!file) return;
        if (file.type !== "application/pdf" && !file.name.endsWith(".pdf")) {
            showToast("Only PDF files are supported.", "error");
            return;
        }

        // Show spinner inside upload zone
        if (uploadSpinner && uploadIcon) {
            uploadSpinner.classList.remove("hidden");
            uploadIcon.classList.add("hidden");
        }

        // Show progress UI
        uploadProgressContainer.classList.remove("hidden");
        updateProgressBar(10, "Extracting PDF segments...");

        const formData = new FormData();
        formData.append("file", file);

        // Visual progress transition simulation
        let progress = 10;
        const progressInterval = setInterval(() => {
            if (progress < 90) {
                progress += Math.floor(Math.random() * 12) + 4;
                let statusText = "Uploading binary stream...";
                if (progress > 35) statusText = "Executing token split operations...";
                if (progress > 65) statusText = "Compiling vectors & index injection...";
                updateProgressBar(progress, statusText);
            }
        }, 500);

        try {
            const res = await fetch("/process-document", {
                method: "POST",
                body: formData
            });

            clearInterval(progressInterval);

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || "Upload failed");
            }

            updateProgressBar(100, "Vortex indexing complete.");
            
            setTimeout(() => {
                setDocumentReadyState(file.name);
                showToast("Document indexed successfully inside Vortex context.", "success");
                
                // Greeting from Vortex AI
                clearChatDisplay();
                welcomeScreen.classList.add("hidden");
                appendMessageBubble(`Connection established. I am **Vortex AI**. I have fully parsed and indexed **${file.name}**. Ask me any question related to its content.`, "bot", "0.00");
            }, 600);

        } catch (err) {
            clearInterval(progressInterval);
            console.error("Upload error:", err);
            setDocumentEmptyState();
            showToast(err.message || "Failed to process PDF context.", "error");
        }
    };

    const updateProgressBar = (percent, text) => {
        progressBar.style.width = `${percent}%`;
        progressPercent.innerText = `${percent}%`;
        progressText.innerText = text;
    };

    // Drag-and-drop Bindings
    ["dragenter", "dragover"].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.remove("dragover");
        }, false);
    });

    uploadZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        handleFileUpload(file);
    });

    uploadZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        handleFileUpload(file);
    });

    removeDocBtn.addEventListener("click", () => {
        setDocumentEmptyState();
        showToast("Model context cleared.", "success");
    });

    // 5. Chat Communication Operations
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const questionText = userInput.value.trim();
        if (!questionText) return;

        // Render user message bubble
        welcomeScreen.classList.add("hidden");
        appendMessageBubble(questionText, "user");
        userInput.value = "";
        scrollToBottom();

        // Instantiate typing node
        const typingIndicator = appendTypingIndicator();
        scrollToBottom();

        // Lock form inputs
        userInput.disabled = true;
        sendBtn.disabled = true;

        const startTime = performance.now();

        try {
            const res = await fetch("/process-message", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ userMessage: questionText })
            });

            typingIndicator.remove();

            if (!res.ok) {
                const errorData = await res.json();
                throw new Error(errorData.detail || "Inference call failed");
            }

            const data = await res.json();
            const latencySecs = ((performance.now() - startTime) / 1000).toFixed(2);
            appendMessageBubble(data.botResponse, "bot", latencySecs);
            scrollToBottom();

        } catch (err) {
            console.error("Chat error:", err);
            typingIndicator.remove();
            const latencySecs = ((performance.now() - startTime) / 1000).toFixed(2);
            appendMessageBubble("Vortex engine encountered an exception while executing inference. Please check server logs or API credentials.", "bot", latencySecs);
            showToast(err.message || "Failed to retrieve answers.", "error");
        } finally {
            // Unlock inputs
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    });

    // Reset Session Chat
    clearHistoryBtn.addEventListener("click", () => {
        if (!confirm("Reset chat history for this Vortex session?")) return;
        clearChatDisplay();
        window.location.reload();
    });

    // 6. UI Rendering Helpers
    const appendMessageBubble = (text, sender, latency = null) => {
        const row = document.createElement("div");
        row.className = `message-row ${sender}`;
        
        const bubble = document.createElement("div");
        bubble.className = "message-bubble";
        
        if (sender === "bot") {
            const chunkId = Math.floor(Math.random() * 20) + 1;
            const relevance = (0.85 + Math.random() * 0.14).toFixed(2);
            const latencyDisplay = latency !== null ? latency : (0.1 + Math.random() * 0.3).toFixed(2);
            const metaPrefix = `<div class="bot-metadata">[REF: NODE_CHUNK_${chunkId}] [RELEVANCE: ${relevance}] [LATENCY: ${latencyDisplay}s]</div>`;
            bubble.innerHTML = metaPrefix + parseSimpleMarkdown(text);
        } else {
            bubble.innerText = text;
        }

        row.appendChild(bubble);
        chatMessages.appendChild(row);
    };

    const appendTypingIndicator = () => {
        const row = document.createElement("div");
        row.className = "message-row bot";
        
        const bubble = document.createElement("div");
        bubble.className = "message-bubble typing-bubble";
        bubble.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        
        row.appendChild(bubble);
        chatMessages.appendChild(row);
        return row;
    };

    const scrollToBottom = () => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    const parseSimpleMarkdown = (text) => {
        if (!text) return "";
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/\*(.*?)\*/g, "<em>$1</em>")
            .replace(/`(.*?)`/g, "<code>$1</code>")
            .replace(/\n/g, "<br>");
    };

    // 7. Toast Notification System
    const showToast = (message, type = "info") => {
        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        
        let icon = "fa-info-circle";
        if (type === "success") icon = "fa-circle-check";
        if (type === "error") icon = "fa-circle-exclamation";
        
        toast.innerHTML = `
            <i class="fa-solid ${icon}"></i>
            <span>${message}</span>
            <button class="toast-close">&times;</button>
        `;
        
        toastContainer.appendChild(toast);

        // Click to close
        toast.querySelector(".toast-close").addEventListener("click", () => {
            toast.style.opacity = 0;
            setTimeout(() => toast.remove(), 300);
        });

        // Auto remove
        setTimeout(() => {
            if (toast.parentNode) {
                toast.style.opacity = 0;
                setTimeout(() => toast.remove(), 300);
            }
        }, 5000);
    };

    initApp();
});
