package com.boot.entity;

import java.io.Serializable;
import java.util.Date;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.Table;
import javax.persistence.Transient;

import lombok.Data;
/**
 * 看板
 */
@Entity
@Table(name = "t_team")
@Data
public class Board implements Serializable{
	private static final long serialVersionUID = 874310660870093440L;
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private Integer id;
	private String name;
	
	@Column(name="user_id")
	private Integer userId;//
	
	@Column(name="team_id")
	private Integer teamId;//
	
	private String description;
	
	@Column(name="create_time")
	private Date createTime = new Date();//创建时间
	
	@Transient
	private String teamName;
}
