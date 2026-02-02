// vLLM CI/CD Pipeline for DGX Spark 2 (SM121/GB10)
// ================================================
// Builds and tests vLLM on spark2 (ARM64 + 8x A100)
// Jenkins master on node5 SSHs to spark2 for execution
//
// Target: NVIDIA GB10 Blackwell (SM121) + ARM64 + CUDA 13.1
// Model: Qwen3-Next-80B-A3B-FP8

pipeline {
    agent any
    
    options {
        timestamps()
        ansiColor('xterm')
        buildDiscarder(logRotator(numToKeepStr: '20', artifactNumToKeepStr: '5'))
        timeout(time: 180, unit: 'MINUTES')
        disableConcurrentBuilds()
    }
    
    environment {
        // Spark2 connection (ARM64 + GPU build server)
        SPARK2_HOST = '192.168.4.208'
        SPARK2_USER = 'seli'
        
        // vLLM server endpoint
        VLLM_API_URL = 'http://192.168.4.208:8000'
        
        // Remote paths on spark2
        VLLM_DIR = '/data/vllm'
        VLLM_ENV = '/data/vllm-env'
        CONTAINER_DIR = '/data/vllm-container'
        
        // Local paths for results
        RESULTS_DIR = 'test-results'
        ALLURE_RESULTS = 'test-results/allure-results'
        
        // Build settings
        TORCH_CUDA_ARCH_LIST = '12.1'
        MAX_JOBS = '20'
    }
    
    parameters {
        booleanParam(
            name: 'REBUILD_VLLM',
            defaultValue: false,
            description: 'Rebuild vLLM from source (preserves torch)'
        )
        booleanParam(
            name: 'REBUILD_FLASHINFER',
            defaultValue: false,
            description: 'Rebuild FlashInfer from source'
        )
        booleanParam(
            name: 'REBUILD_DOCKER',
            defaultValue: false,
            description: 'Rebuild Docker container image'
        )
        booleanParam(
            name: 'RUN_BENCHMARKS',
            defaultValue: false,
            description: 'Run performance benchmarks'
        )
        booleanParam(
            name: 'SYNC_CODE',
            defaultValue: true,
            description: 'Pull latest code on spark2 before build'
        )
        string(
            name: 'BENCHMARK_PROMPTS',
            defaultValue: '50',
            description: 'Number of prompts for benchmark'
        )
    }
    
    stages {
        stage('Prepare') {
            steps {
                echo '📁 Preparing test environment...'
                sh """
                    mkdir -p ${RESULTS_DIR}
                    mkdir -p ${ALLURE_RESULTS}
                    rm -f ${RESULTS_DIR}/*.xml ${RESULTS_DIR}/*.html 2>/dev/null || true
                    rm -rf ${ALLURE_RESULTS}/* 2>/dev/null || true
                """
            }
        }
        
        stage('Spark2 Health Check') {
            steps {
                echo '🔍 Checking spark2 connectivity and GPU status...'
                sshagent(['ssh-credentials']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                            echo '=== System Info ==='
                            uname -a
                            echo ''
                            echo '=== GPU Status ==='
                            nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
                            echo ''
                            echo '=== Docker Status ==='
                            docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}' | head -10
                            echo ''
                            echo '=== Disk Space ==='
                            df -h /data
                        "
                    '''
                }
            }
        }
        
        stage('vLLM Server Health') {
            steps {
                echo '🔍 Checking vLLM server status...'
                script {
                    def health = sh(
                        script: "curl -sf ${VLLM_API_URL}/health && echo 'OK' || echo 'DOWN'",
                        returnStdout: true
                    ).trim()
                    
                    def models = sh(
                        script: "curl -sf ${VLLM_API_URL}/v1/models | jq -r '.data[0].id' 2>/dev/null || echo 'UNKNOWN'",
                        returnStdout: true
                    ).trim()
                    
                    if (health.contains('OK')) {
                        echo "✅ vLLM Server: Running"
                        echo "📦 Loaded Model: ${models}"
                    } else {
                        echo "⚠️ vLLM Server: Not responding (may need to start)"
                    }
                }
            }
        }
        
        stage('Sync Code') {
            when {
                expression { params.SYNC_CODE == true }
            }
            steps {
                echo '📥 Syncing latest code to spark2...'
                sshagent(['ssh-credentials']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                            cd ${VLLM_DIR} && \
                            git fetch origin && \
                            git status && \
                            git pull origin \\$(git branch --show-current) && \
                            git log -1 --pretty=format:'%h - %s (%an, %ar)'
                        "
                    '''
                }
            }
        }
        
        stage('Verify Torch') {
            steps {
                echo '🔧 Verifying CUDA torch installation...'
                sshagent(['ssh-credentials']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                            source ${VLLM_ENV}/bin/activate && \
                            python3 -c \\"
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
\\"
                        "
                    '''
                }
            }
        }
        
        stage('Build FlashInfer') {
            when {
                expression { params.REBUILD_FLASHINFER == true }
            }
            steps {
                echo '🔨 Rebuilding FlashInfer on spark2...'
                sshagent(['ssh-credentials']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                            cd ${CONTAINER_DIR} && \
                            ./safe-rebuild-vllm.sh --flashinfer-only --clear-cache -y
                        "
                    '''
                }
            }
        }
        
        stage('Build vLLM') {
            when {
                expression { params.REBUILD_VLLM == true }
            }
            steps {
                echo '🔨 Rebuilding vLLM on spark2 (preserving torch)...'
                sshagent(['ssh-credentials']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                            cd ${CONTAINER_DIR} && \
                            ./safe-rebuild-vllm.sh --vllm-only -y
                        "
                    '''
                }
            }
        }
        
        stage('Build Docker') {
            when {
                expression { params.REBUILD_DOCKER == true }
            }
            steps {
                echo '🐳 Rebuilding Docker container on spark2...'
                sshagent(['ssh-credentials']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                            cd ${CONTAINER_DIR} && \
                            ./safe-rebuild-vllm.sh --sync --docker -y 2>&1 | tee build.log
                        "
                    '''
                }
            }
        }
        
        stage('Unit Tests') {
            steps {
                echo '🧪 Running unit tests on spark2...'
                sshagent(['ssh-credentials']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                            source ${VLLM_ENV}/bin/activate && \
                            cd ${VLLM_DIR} && \
                            mkdir -p test-results/allure-results && \
                            python -m pytest tests/test_config.py tests/test_logger.py tests/test_utils.py \
                                -v --tb=short \
                                --junitxml=test-results/unit_results.xml \
                                --alluredir=test-results/allure-results \
                                --timeout=120 \
                                -x || true
                        "
                        
                        # Copy results back to Jenkins
                        scp -o StrictHostKeyChecking=no \
                            ${SPARK2_USER}@${SPARK2_HOST}:${VLLM_DIR}/test-results/*.xml \
                            ${RESULTS_DIR}/ 2>/dev/null || true
                        scp -o StrictHostKeyChecking=no -r \
                            ${SPARK2_USER}@${SPARK2_HOST}:${VLLM_DIR}/test-results/allure-results/* \
                            ${ALLURE_RESULTS}/ 2>/dev/null || true
                    '''
                }
            }
            post {
                always {
                    junit(
                        testResults: "${RESULTS_DIR}/unit_results.xml",
                        allowEmptyResults: true,
                        skipPublishingChecks: true
                    )
                }
            }
        }
        
        stage('API Health Tests') {
            steps {
                echo '🌐 Running API tests against live vLLM server...'
                script {
                    // Test health endpoint
                    def healthResult = sh(
                        script: """
                            curl -sf ${VLLM_API_URL}/health && echo 'PASS' || echo 'FAIL'
                        """,
                        returnStdout: true
                    ).trim()
                    
                    // Test models endpoint
                    def modelsResult = sh(
                        script: """
                            curl -sf ${VLLM_API_URL}/v1/models | jq -e '.data | length > 0' && echo 'PASS' || echo 'FAIL'
                        """,
                        returnStdout: true
                    ).trim()
                    
                    // Quick inference test
                    def inferResult = sh(
                        script: '''
                            curl -sf -X POST "${VLLM_API_URL}/v1/chat/completions" \
                                -H "Content-Type: application/json" \
                                -d '{"model":"/models/Qwen3-Next-80B-A3B-FP8","messages":[{"role":"user","content":"Say hello"}],"max_tokens":10}' \
                                | jq -e '.choices[0].message.content' && echo 'PASS' || echo 'FAIL'
                        ''',
                        returnStdout: true
                    ).trim()
                    
                    echo "Health Check: ${healthResult.contains('PASS') ? '✅' : '❌'}"
                    echo "Models Check: ${modelsResult.contains('PASS') ? '✅' : '❌'}"
                    echo "Inference Check: ${inferResult.contains('PASS') ? '✅' : '❌'}"
                    
                    // Write JUnit result
                    writeFile file: "${RESULTS_DIR}/api_results.xml", text: """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="vLLM API Tests" tests="3" failures="${[healthResult, modelsResult, inferResult].count { it.contains('FAIL') }}">
    <testcase classname="api" name="health_check" time="1"><${healthResult.contains('PASS') ? '/' : 'failure message="Health check failed"/'}></testcase>
    <testcase classname="api" name="models_list" time="1"><${modelsResult.contains('PASS') ? '/' : 'failure message="Models list failed"/'}></testcase>
    <testcase classname="api" name="inference" time="5"><${inferResult.contains('PASS') ? '/' : 'failure message="Inference failed"/'}></testcase>
</testsuite>"""
                }
            }
            post {
                always {
                    junit(
                        testResults: "${RESULTS_DIR}/api_results.xml",
                        allowEmptyResults: true,
                        skipPublishingChecks: true
                    )
                }
            }
        }
        
        stage('Benchmarks') {
            when {
                anyOf {
                    expression { params.RUN_BENCHMARKS == true }
                    triggeredBy 'TimerTrigger'
                }
            }
            steps {
                echo "⚡ Running performance benchmarks (${params.BENCHMARK_PROMPTS} prompts)..."
                sshagent(['ssh-credentials']) {
                    sh '''
                        ssh -o StrictHostKeyChecking=no ${SPARK2_USER}@${SPARK2_HOST} "
                            source ${VLLM_ENV}/bin/activate && \
                            cd ${VLLM_DIR} && \
                            python benchmarks/benchmark_serving.py \
                                --backend vllm \
                                --base-url ${VLLM_API_URL} \
                                --model /models/Qwen3-Next-80B-A3B-FP8 \
                                --num-prompts ${BENCHMARK_PROMPTS} \
                                --request-rate 1.0 \
                                --save-result \
                                --result-dir test-results \
                                --result-filename benchmark_serving.json || true
                        "
                        
                        # Copy benchmark results
                        scp -o StrictHostKeyChecking=no \
                            ${SPARK2_USER}@${SPARK2_HOST}:${VLLM_DIR}/test-results/benchmark_*.json \
                            ${RESULTS_DIR}/ 2>/dev/null || true
                    '''
                }
            }
            post {
                always {
                    archiveArtifacts(
                        artifacts: "${RESULTS_DIR}/benchmark_*.json",
                        allowEmptyArchive: true
                    )
                }
            }
        }
        
        stage('Allure Report') {
            steps {
                echo '📈 Generating Allure report...'
                script {
                    try {
                        allure([
                            includeProperties: false,
                            jdk: '',
                            properties: [],
                            reportBuildPolicy: 'ALWAYS',
                            results: [[path: "${ALLURE_RESULTS}"]]
                        ])
                    } catch (e) {
                        echo "Allure report generation skipped: ${e.message}"
                    }
                }
            }
        }
    }
    
    post {
        always {
            echo '🧹 Archiving results...'
            archiveArtifacts(
                artifacts: "${RESULTS_DIR}/**/*",
                allowEmptyArchive: true
            )
        }
        success {
            echo '''
╔══════════════════════════════════════════════════════════════╗
║  ✅ vLLM Pipeline completed successfully!                    ║
║                                                              ║
║  Server: http://192.168.4.208:8000                          ║
║  Model:  Qwen3-Next-80B-A3B-FP8                             ║
║  GPU:    NVIDIA GB10 (SM121) - 115GB                        ║
╚══════════════════════════════════════════════════════════════╝
'''
        }
        failure {
            echo '❌ vLLM Pipeline failed! Check logs for details.'
        }
        unstable {
            echo '⚠️ vLLM Pipeline unstable (some tests failed)'
        }
    }
}

