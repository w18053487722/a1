#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import torch
import numpy as np
import random
import os
import logging
import argparse
import difflib
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子以确保可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 仇恨言论数据集类
class HateSpeechDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_prefix = "抽取中文仇恨言论四元组: "
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        content = self.task_prefix + item["content"]
        
        # 训练数据有输出标签
        if "output" in item:
            output = item["output"]
        else:
            output = ""  # 测试数据没有标签
        
        # 对输入进行编码
        input_encoding = self.tokenizer.encode_plus(
            content,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 仅对训练数据的输出进行编码
        if output:
            output_encoding = self.tokenizer.encode_plus(
                output,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = input_encoding["input_ids"].squeeze()
            attention_mask = input_encoding["attention_mask"].squeeze()
            labels = output_encoding["input_ids"].squeeze()
            
            # 将填充token的id替换为-100，在计算损失时忽略它们
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "id": item["id"]
            }
        else:
            input_ids = input_encoding["input_ids"].squeeze()
            attention_mask = input_encoding["attention_mask"].squeeze()
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "id": item["id"]
            }

# 从JSON文件加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 训练模型的函数
def train(model, tokenizer, train_dataloader, optimizer, scheduler, device, epochs, gradient_accumulation_steps=1):
    model.train()
    
    for epoch in range(epochs):
        logger.info(f"开始第 {epoch+1}/{epochs} 轮训练")
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            # 将批次移至设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 对梯度累积进行损失归一化
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item()
            
            # 更新权重
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": total_loss / (step + 1)})
        
        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"第 {epoch+1} 轮完成。平均训练损失: {avg_train_loss:.4f}")
        
        # 每轮结束后保存模型
        output_dir = f"model_epoch_{epoch+1}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"模型已保存到 {output_dir}")

# 用于预测测试数据输出的函数
def predict(model, tokenizer, test_dataloader, device, max_length=512):
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="预测中"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ids = batch["id"]
            
            # 生成预测
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            
            # 解码预测
            decoded_preds = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            
            # 格式化预测，确保它们符合要求的格式
            formatted_preds = []
            for pred in decoded_preds:
                # 检查预测是否已经遵循正确的格式
                if not pred.endswith(" [END]"):
                    if " [SEP] " in pred:
                        # 多个四元组
                        quads = pred.split(" [SEP] ")
                        formatted_quads = []
                        for i, quad in enumerate(quads):
                            if i == len(quads) - 1:
                                # 最后一个四元组应该以[END]结尾
                                if not quad.endswith(" [END]"):
                                    formatted_quads.append(quad + " [END]")
                                else:
                                    formatted_quads.append(quad)
                            else:
                                formatted_quads.append(quad)
                        formatted_pred = " [SEP] ".join(formatted_quads)
                    else:
                        # 单个四元组
                        if not pred.endswith(" [END]"):
                            formatted_pred = pred + " [END]"
                        else:
                            formatted_pred = pred
                else:
                    formatted_pred = pred
                
                formatted_preds.append(formatted_pred)
            
            # 存储预测
            for id_val, pred in zip(ids, formatted_preds):
                predictions[id_val.item()] = pred
    
    return predictions

# 将预测保存到文件
def save_predictions(predictions, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for id_val in sorted(predictions.keys()):
            f.write(f"{predictions[id_val]}\n")

# 计算字符串相似度
def calculate_similarity(pred_str, gold_str):
    sequence_matcher = difflib.SequenceMatcher(None, pred_str, gold_str)
    m = sequence_matcher.find_longest_match(0, len(pred_str), 0, len(gold_str)).size
    similarity = (2 * m) / (len(pred_str) + len(gold_str)) if (len(pred_str) + len(gold_str)) > 0 else 0
    return similarity

# 评估函数
def evaluate(pred_file, gold_file):
    # 加载预测和黄金标准
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_lines = f.readlines()
    
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_lines = f.readlines()
    
    if len(pred_lines) != len(gold_lines):
        logger.warning(f"预测文件有 {len(pred_lines)} 行，但黄金标准文件有 {len(gold_lines)} 行")
    
    hard_tp, hard_fp, hard_fn = 0, 0, 0
    soft_tp, soft_fp, soft_fn = 0, 0, 0
    
    for pred_line, gold_line in zip(pred_lines, gold_lines):
        pred_line = pred_line.strip()
        gold_line = gold_line.strip()
        
        # 处理行尾的[END]情况
        if pred_line.endswith(" [END]"):
            pred_line = pred_line[:-6]  # 移除尾部的[END]
        
        if gold_line.endswith(" [END]"):
            gold_line = gold_line[:-6]  # 移除尾部的[END]
        
        # 分割为四元组
        pred_quadruples = pred_line.split(" [SEP] ")
        gold_quadruples = gold_line.split(" [SEP] ")
        
        # 处理每个四元组
        pred_set = []
        for quad in pred_quadruples:
            if quad.strip():
                # 每个四元组应该有4个由" | "分隔的元素
                elements = quad.strip().split(" | ")
                if len(elements) == 4:
                    pred_set.append(tuple(elements))
        
        gold_set = []
        for quad in gold_quadruples:
            if quad.strip():
                # 每个四元组应该有4个由" | "分隔的元素
                elements = quad.strip().split(" | ")
                if len(elements) == 4:
                    gold_set.append(tuple(elements))
        
        # 硬匹配
        hard_matched = set()
        for pred_quad in pred_set:
            matched = False
            for i, gold_quad in enumerate(gold_set):
                if i not in hard_matched and pred_quad == gold_quad:
                    hard_tp += 1
                    hard_matched.add(i)
                    matched = True
                    break
            if not matched:
                hard_fp += 1
        
        hard_fn += len(gold_set) - len(hard_matched)
        
        # 软匹配
        soft_matched = set()
        for pred_quad in pred_set:
            if len(pred_quad) != 4:
                continue
                
            matched = False
            for i, gold_quad in enumerate(gold_set):
                if len(gold_quad) != 4:
                    continue
                    
                # 检查目标群体和是否仇恨是否完全匹配
                if pred_quad[2].lower() == gold_quad[2].lower() and pred_quad[3].lower() == gold_quad[3].lower():
                    # 检查评论对象和论点是否相似度>0.5
                    target_sim = calculate_similarity(pred_quad[0], gold_quad[0])
                    arg_sim = calculate_similarity(pred_quad[1], gold_quad[1])
                    
                    if target_sim > 0.5 and arg_sim > 0.5 and i not in soft_matched:
                        soft_tp += 1
                        soft_matched.add(i)
                        matched = True
                        break
            
            if not matched:
                soft_fp += 1
        
        soft_fn += len(gold_set) - len(soft_matched)
    
    # 计算F1分数
    hard_precision = hard_tp / (hard_tp + hard_fp) if (hard_tp + hard_fp) > 0 else 0
    hard_recall = hard_tp / (hard_tp + hard_fn) if (hard_tp + hard_fn) > 0 else 0
    hard_f1 = 2 * hard_precision * hard_recall / (hard_precision + hard_recall) if (hard_precision + hard_recall) > 0 else 0
    
    soft_precision = soft_tp / (soft_tp + soft_fp) if (soft_tp + soft_fp) > 0 else 0
    soft_recall = soft_tp / (soft_tp + soft_fn) if (soft_tp + soft_fn) > 0 else 0
    soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0
    
    avg_f1 = (hard_f1 + soft_f1) / 2
    
    results = {
        "hard_precision": hard_precision,
        "hard_recall": hard_recall,
        "hard_f1": hard_f1,
        "soft_precision": soft_precision,
        "soft_recall": soft_recall,
        "soft_f1": soft_f1,
        "avg_f1": avg_f1
    }
    
    return results

# 主函数
def main():
    parser = argparse.ArgumentParser(description="训练和评估中文仇恨言论检测模型")
    parser.add_argument("--train_file", default="train.json", type=str, help="训练数据路径")
    parser.add_argument("--test_file", default="test.json", type=str, help="测试数据路径")
    parser.add_argument("--model_name", default="Langboat/mengzi-t5-base", type=str, help="预训练模型路径")
    parser.add_argument("--output_dir", default="output", type=str, help="输出目录")
    parser.add_argument("--prediction_file", default="predictions.txt", type=str, help="预测输出文件")
    parser.add_argument("--gold_file", default="gold.txt", type=str, help="评估用的黄金标准文件")
    parser.add_argument("--max_length", default=512, type=int, help="最大序列长度")
    parser.add_argument("--batch_size", default=2, type=int, help="训练和评估的批次大小")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="权重衰减")
    parser.add_argument("--epochs", default=10, type=int, help="训练轮数")
    parser.add_argument("--warmup_steps", default=0, type=int, help="预热步数")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="梯度累积步数")
    parser.add_argument("--seed", default=42, type=int, help="随机种子")
    parser.add_argument("--do_train", action="store_true", help="是否运行训练")
    parser.add_argument("--do_predict", action="store_true", help="是否运行预测")
    parser.add_argument("--do_eval", action="store_true", help="是否运行评估")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 如果输出目录不存在，则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载分词器和模型
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)
    
    # 训练
    if args.do_train:
        logger.info("加载训练数据...")
        train_data = load_data(args.train_file)
        train_dataset = HateSpeechDataset(train_data, tokenizer, args.max_length)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        # 准备优化器和调度器
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
        )
        
        logger.info("开始训练...")
        train(model, tokenizer, train_dataloader, optimizer, scheduler, device, args.epochs, args.gradient_accumulation_steps)
    
    # 预测
    if args.do_predict:
        logger.info("加载测试数据...")
        test_data = load_data(args.test_file)
        test_dataset = HateSpeechDataset(test_data, tokenizer, args.max_length)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        logger.info("开始预测...")
        predictions = predict(model, tokenizer, test_dataloader, device, args.max_length)
        
        prediction_file = os.path.join(args.output_dir, args.prediction_file)
        save_predictions(predictions, prediction_file)
        logger.info(f"预测已保存到 {prediction_file}")
    
    # 评估
    if args.do_eval:
        logger.info("开始评估...")
        prediction_file = os.path.join(args.output_dir, args.prediction_file)
        results = evaluate(prediction_file, args.gold_file)
        
        logger.info(f"硬匹配准确率: {results['hard_precision']:.4f}")
        logger.info(f"硬匹配召回率: {results['hard_recall']:.4f}")
        logger.info(f"硬匹配F1: {results['hard_f1']:.4f}")
        logger.info(f"软匹配准确率: {results['soft_precision']:.4f}")
        logger.info(f"软匹配召回率: {results['soft_recall']:.4f}")
        logger.info(f"软匹配F1: {results['soft_f1']:.4f}")
        logger.info(f"平均F1: {results['avg_f1']:.4f}")
        
        # 将结果保存到文件
        results_file = os.path.join(args.output_dir, "results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"结果已保存到 {results_file}")

if __name__ == "__main__":
    main()