"""
创建示例翻译数据集
基于Multi30k数据集的特点，创建一个适合学习的小数据集
"""

import json
import os
from typing import List, Tuple


def create_sample_dataset():
    """
    创建示例翻译数据集
    基于Multi30k数据集的特点：
    - 图像描述翻译
    - 英语-德语
    - 句子长度适中（5-20个词）
    - 包含日常场景描述
    """
    
    # 示例数据：英语-德语图像描述翻译
    sample_data = [
        # 人物描述
        ("A man is standing in front of a building", "Ein Mann steht vor einem Gebäude"),
        ("A woman is walking on the street", "Eine Frau geht auf der Straße"),
        ("A child is playing in the park", "Ein Kind spielt im Park"),
        ("A group of people are talking", "Eine Gruppe von Menschen redet"),
        ("A young couple is holding hands", "Ein junges Paar hält Hände"),
        
        # 动物描述
        ("A dog is running in the field", "Ein Hund läuft auf dem Feld"),
        ("A cat is sleeping on the sofa", "Eine Katze schläft auf dem Sofa"),
        ("A bird is flying in the sky", "Ein Vogel fliegt am Himmel"),
        ("A horse is standing in the stable", "Ein Pferd steht im Stall"),
        ("A fish is swimming in the water", "Ein Fisch schwimmt im Wasser"),
        
        # 交通工具
        ("A car is driving on the road", "Ein Auto fährt auf der Straße"),
        ("A bus is stopping at the station", "Ein Bus hält an der Station"),
        ("A train is leaving the platform", "Ein Zug verlässt den Bahnsteig"),
        ("A plane is flying over the city", "Ein Flugzeug fliegt über die Stadt"),
        ("A bicycle is parked near the tree", "Ein Fahrrad ist neben dem Baum geparkt"),
        
        # 自然场景
        ("The sun is shining brightly", "Die Sonne scheint hell"),
        ("It is raining heavily outside", "Es regnet stark draußen"),
        ("The wind is blowing the leaves", "Der Wind bläst die Blätter"),
        ("Snow is falling from the sky", "Schnee fällt vom Himmel"),
        ("The moon is visible in the night", "Der Mond ist in der Nacht sichtbar"),
        
        # 室内场景
        ("A table is set for dinner", "Ein Tisch ist für das Abendessen gedeckt"),
        ("A book is lying on the desk", "Ein Buch liegt auf dem Schreibtisch"),
        ("A lamp is lighting the room", "Eine Lampe beleuchtet den Raum"),
        ("A chair is placed near the window", "Ein Stuhl steht am Fenster"),
        ("A picture is hanging on the wall", "Ein Bild hängt an der Wand"),
        
        # 食物描述
        ("A pizza is being prepared", "Eine Pizza wird zubereitet"),
        ("Fresh fruits are on the table", "Frische Früchte sind auf dem Tisch"),
        ("A cup of coffee is steaming", "Eine Tasse Kaffee dampft"),
        ("Bread is baking in the oven", "Brot backt im Ofen"),
        ("A salad is being served", "Ein Salat wird serviert"),
        
        # 活动描述
        ("People are dancing at the party", "Menschen tanzen auf der Party"),
        ("A game is being played", "Ein Spiel wird gespielt"),
        ("Music is playing in the background", "Musik spielt im Hintergrund"),
        ("A movie is being watched", "Ein Film wird angeschaut"),
        ("A song is being sung", "Ein Lied wird gesungen"),
        
        # 建筑和地点
        ("A church is standing in the center", "Eine Kirche steht im Zentrum"),
        ("A school is located on the hill", "Eine Schule liegt auf dem Hügel"),
        ("A hospital is near the station", "Ein Krankenhaus ist nahe der Station"),
        ("A library is open for visitors", "Eine Bibliothek ist für Besucher geöffnet"),
        ("A museum is showing new exhibits", "Ein Museum zeigt neue Ausstellungen"),
        
        # 时间描述
        ("It is morning and people are waking up", "Es ist Morgen und Menschen wachen auf"),
        ("It is noon and the sun is high", "Es ist Mittag und die Sonne steht hoch"),
        ("It is evening and lights are turning on", "Es ist Abend und Lichter gehen an"),
        ("It is night and stars are visible", "Es ist Nacht und Sterne sind sichtbar"),
        ("It is weekend and people are relaxing", "Es ist Wochenende und Menschen entspannen"),
        
        # 情感和状态
        ("Everyone is happy and smiling", "Alle sind glücklich und lächeln"),
        ("The atmosphere is peaceful and quiet", "Die Atmosphäre ist friedlich und ruhig"),
        ("People are busy with their work", "Menschen sind mit ihrer Arbeit beschäftigt"),
        ("The mood is cheerful and lively", "Die Stimmung ist fröhlich und lebhaft"),
        ("Everyone is excited about the event", "Alle sind aufgeregt über das Ereignis")
    ]
    
    return sample_data


def save_dataset(data: List[Tuple[str, str]], output_dir: str = "data"):
    """保存数据集到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 分离源语言和目标语言
    src_data = [item[0] for item in data]
    tgt_data = [item[1] for item in data]
    
    # 保存为文本文件
    with open(os.path.join(output_dir, "train.en"), 'w', encoding='utf-8') as f:
        f.write('\\n'.join(src_data))
    
    with open(os.path.join(output_dir, "train.de"), 'w', encoding='utf-8') as f:
        f.write('\\n'.join(tgt_data))
    
    # 保存为JSON格式
    json_data = [{"src": src, "tgt": tgt} for src, tgt in data]
    with open(os.path.join(output_dir, "train.json"), 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据集已保存到 {output_dir} 目录")
    print(f"包含 {len(data)} 个句子对")


def analyze_dataset(data: List[Tuple[str, str]]):
    """分析数据集特点"""
    src_lengths = [len(src.split()) for src, _ in data]
    tgt_lengths = [len(tgt.split()) for tgt, _ in data]
    
    print("\\n=== 数据集分析 ===")
    print(f"总句子对数: {len(data)}")
    print(f"源语言平均长度: {sum(src_lengths) / len(src_lengths):.2f} 词")
    print(f"目标语言平均长度: {sum(tgt_lengths) / len(tgt_lengths):.2f} 词")
    print(f"源语言最大长度: {max(src_lengths)} 词")
    print(f"目标语言最大长度: {max(tgt_lengths)} 词")
    print(f"源语言最小长度: {min(src_lengths)} 词")
    print(f"目标语言最小长度: {min(tgt_lengths)} 词")
    
    # 词汇统计
    src_vocab = set()
    tgt_vocab = set()
    
    for src, tgt in data:
        src_vocab.update(src.lower().split())
        tgt_vocab.update(tgt.lower().split())
    
    print(f"\\n源语言词汇数: {len(src_vocab)}")
    print(f"目标语言词汇数: {len(tgt_vocab)}")
    
    # 显示一些示例
    print("\\n=== 示例句子 ===")
    for i in range(min(5, len(data))):
        print(f"{i+1}. EN: {data[i][0]}")
        print(f"   DE: {data[i][1]}")


if __name__ == "__main__":
    # 创建数据集
    data = create_sample_dataset()
    
    # 分析数据集
    analyze_dataset(data)
    
    # 保存数据集
    save_dataset(data)
    
    print("\\n✅ 示例数据集创建完成！")
    print("\\n数据集特点：")
    print("- 基于Multi30k数据集特点")
    print("- 英语-德语翻译")
    print("- 图像描述场景")
    print("- 句子长度适中（5-20词）")
    print("- 包含日常场景和活动")
    print("- 适合学习和测试Transformer模型")
